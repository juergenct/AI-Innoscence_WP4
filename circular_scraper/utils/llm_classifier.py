# circular_scraper/utils/llm_classifier.py
"""
Local LLM classification via Ollama HTTP API.

Provides a unified classifier for:
- Hamburg relevance
- Circular Economy (CE) relevance
- CE ecosystem role

Model: default 'qwen3-4b-instruct' (created in Ollama from
Hugging Face model 'Qwen/Qwen3-4B-Instruct-2507').
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import time
import pandas as pd
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    entity_id: str
    hamburg_related: bool
    ce_related: bool
    role: str
    confidence: float


class LLMClassifier:
    def __init__(self,
                 server_url: str = 'http://localhost:11434',
                 model: str = 'qwen3-4b-instruct',
                 timeout: int = 30,
                 connect_timeout: int = 5,
                 max_workers: int = 2,
                 debug_dir: Optional[str] = None):
        self.base = server_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_workers = max_workers
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def check_server(self) -> bool:
        try:
            resp = requests.get(f"{self.base}/api/tags", timeout=(self.connect_timeout, 10))
            resp.raise_for_status()
            models = [m.get('name') for m in resp.json().get('models', [])]
            if self.model not in models:
                logger.warning(f"Model '{self.model}' not listed by Ollama; available: {models}")
            return True
        except Exception as e:
            logger.error(f"Cannot reach Ollama at {self.base}: {e}")
            return False

    def _build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Construct chat messages for Ollama /api/chat.

        Expects context with keys:
          - entity_id, entity_name, entity_root_url, domain
          - samples: list of dicts with {url, title, meta_description, extracted_text}
        """
        system = (
            "You are a precise classifier. Analyze the provided web snippets and return a strict JSON object with keys: "
            "hamburg_related (bool), ce_related (bool), role (string), confidence (0-1). "
            "Hamburg-related means the organization or page is connected to Hamburg (Germany) via address, activities, or explicit mentions. "
            "Circular Economy (CE) relevance includes recycling, resource efficiency, waste management, reuse, bioeconomy, sustainability, etc. "
            "\nRole MUST be exactly one of these categories based on the entity's primary function:\n"
            "- Students: Student organizations, student initiatives, student councils\n"
            "- Researchers: Individual researchers, research groups, research departments\n" 
            "- Higher Education Institutions: Universities, colleges, academies\n"
            "- Research Institutes: Dedicated research centers, institutes, laboratories\n"
            "- Non-Governmental Organizations: NGOs, non-profits, associations, foundations\n"
            "- Industry Partners: Companies, manufacturers, service providers, consultancies\n"
            "- Startups and Entrepreneurs: New ventures, entrepreneurial initiatives, incubators\n"
            "- Public Authorities: Government bodies, municipal services, regulatory agencies\n"
            "- Policy Makers: Legislative bodies, policy institutes, advisory councils\n"
            "- End-Users: Consumer groups, user communities, citizen platforms\n"
            "- Citizen Associations: Community groups, neighborhood initiatives, civic organizations\n"
            "- Media and Communication Partners: News outlets, PR agencies, communication platforms\n"
            "- Funding Bodies: Grant organizations, investment funds, financial institutions\n"
            "- Knowledge and Innovation Communities: Innovation hubs, knowledge platforms, networks\n"
            "- Unknown: Use only if the entity doesn't fit any category above\n"
            "Select the single most appropriate role. Output JSON only, no additional text."
        )
        user = {
            "entity": {
                "entity_id": context.get("entity_id"),
                "entity_name": context.get("entity_name"),
                "entity_root_url": context.get("entity_root_url"),
                "domain": context.get("domain"),
            },
            "samples": [
                {
                    "url": s.get("url"),
                    "title": s.get("title"),
                    "meta_description": s.get("meta_description"),
                    "excerpt": (s.get("extracted_text") or "")[:400],
                }
                for s in (context.get("samples") or [])
            ],
        }
        user_text = (
            "Classify this entity and return JSON only.\n\nContext:\n" + json.dumps(user, ensure_ascii=False)
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

    def _call_ollama(self, messages: List[Dict[str, str]], entity_id: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "num_ctx": 1024,
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 64
            }
        }
        logger.info(f"LLM POST {url} model={self.model}")
        t0 = time.monotonic()
        resp = requests.post(url, json=payload, timeout=(self.connect_timeout, self.timeout))
        resp.raise_for_status()
        data = resp.json()
        dt = time.monotonic() - t0
        logger.info(f"LLM response {len(resp.content)} bytes in {dt:.2f}s")

        # Debug dump
        if self.debug_dir:
            ts = int(time.time())
            base = f"{entity_id or 'entity'}_{ts}"
            try:
                with open(self.debug_dir / f"{base}_request.json", 'w', encoding='utf-8') as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                with open(self.debug_dir / f"{base}_response.json", 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write LLM debug files: {e}")
        content = data.get("message", {}).get("content", "").strip()
        # Try to parse JSON from content directly
        try:
            # Some models may wrap JSON in code fences, extract the JSON object region
            start = content.find('{')
            end = content.rfind('}')
            content_slice = content[start:end+1] if (start != -1 and end != -1 and end > start) else content
            return json.loads(content_slice)
        except Exception:
            logger.warning("LLM returned non-JSON content; defaulting to Unknown")
            return {
                "hamburg_related": False,
                "ce_related": False,
                "role": "Unknown",
                "confidence": 0.0,
            }

    def classify_entity(self, context: Dict[str, Any]) -> LLMResult:
        msgs = self._build_prompt(context)
        result = self._call_ollama(msgs, entity_id=context.get("entity_id"))
        return LLMResult(
            entity_id=context.get("entity_id", ""),
            hamburg_related=bool(result.get("hamburg_related", False)),
            ce_related=bool(result.get("ce_related", False)),
            role=str(result.get("role", "Unknown"))[:100],
            confidence=float(result.get("confidence", 0.0)),
        )

    def classify_from_entities_df(self,
                                  entities_df: pd.DataFrame,
                                  sample_per_entity: int = 3,
                                  max_entities: int | None = None) -> pd.DataFrame:
        """Group by entity_id and classify using a few samples per entity."""
        cols_required = {"entity_id", "entity_name", "entity_root_url", "domain", "url", "title", "meta_description", "extracted_text"}
        for c in cols_required:
            if c not in entities_df.columns:
                entities_df[c] = ""

        # Build contexts (limit first)
        contexts: List[Dict[str, Any]] = []
        count = 0
        for entity_id, group in entities_df.groupby('entity_id', dropna=False):
            ctx = {
                "entity_id": entity_id,
                "entity_name": str(group['entity_name'].iloc[0]) if 'entity_name' in group else '',
                "entity_root_url": str(group['entity_root_url'].iloc[0]) if 'entity_root_url' in group else '',
                "domain": str(group['domain'].iloc[0]) if 'domain' in group else '',
                "samples": group.head(sample_per_entity).to_dict(orient='records'),
            }
            contexts.append(ctx)
            count += 1
            if max_entities is not None and count >= max_entities:
                break

        results: List[LLMResult] = []
        if not self.check_server():
            logger.error("Skipping LLM classification because server is unavailable")
            return pd.DataFrame([r.__dict__ for r in results])

        # Parallel classification with small pool
        logger.info(f"Classifying {len(contexts)} entities | samples={sample_per_entity} | workers={self.max_workers}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_to_ctx = {ex.submit(self.classify_entity, ctx): ctx for ctx in contexts}
            for idx, future in enumerate(as_completed(future_to_ctx), start=1):
                ctx = future_to_ctx[future]
                entity_id = ctx.get('entity_id')
                try:
                    res = future.result(timeout=self.timeout + self.connect_timeout + 5)
                except Exception as e:
                    logger.error(f"LLM classification failed for {entity_id}: {e}")
                    res = LLMResult(entity_id, False, False, "Unknown", 0.0)
                results.append(res)
                logger.info(f"Classified {idx}/{len(contexts)} entities (entity_id={entity_id})")

        df = pd.DataFrame([r.__dict__ for r in results])
        return df

    def merge_into_aggregated(self, aggregated_csv: str, classified_df: pd.DataFrame) -> str:
        agg = pd.read_csv(aggregated_csv) if pd.io.common.file_exists(aggregated_csv) else pd.DataFrame()
        if agg.empty:
            logger.warning("Aggregated entities CSV missing or empty; creating new")
            out = classified_df.rename(columns={
                'hamburg_related': 'hamburg_llm',
                'ce_related': 'ce_llm',
                'role': 'role_llm',
            })
        else:
            out = agg.merge(classified_df, on='entity_id', how='left')
            out.rename(columns={
                'hamburg_related': 'hamburg_llm',
                'ce_related': 'ce_llm',
                'role': 'role_llm',
            }, inplace=True)
        out_path = aggregated_csv.replace('.csv', '_with_llm.csv')
        out.to_csv(out_path, index=False)
        logger.info(f"Wrote LLM-annotated aggregated entities to {out_path}")
        return out_path


