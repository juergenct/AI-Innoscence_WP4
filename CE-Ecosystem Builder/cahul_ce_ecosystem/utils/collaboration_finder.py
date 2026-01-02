"""Collaboration Finder for CE Ecosystem.

This module provides methods to find collaboration opportunities
based on capabilities, needs, clusters, and geographic proximity.

IMPORTANT: All output must be in English.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from math import radians, sin, cos, sqrt, atan2

from cahul_ce_ecosystem.models.schemas import EntityProfileDict, ClusterDict


@dataclass
class CollaborationMatch:
    """Represents a potential collaboration match."""
    entity_name: str
    entity_url: str
    match_type: str  # 'capability', 'need', 'cluster', 'geographic', 'value_chain'
    match_reason: str
    relevance_score: float  # 0.0 - 1.0
    matched_items: List[str] = field(default_factory=list)


@dataclass
class CollaborationOpportunity:
    """A collaboration opportunity between two or more entities."""
    entities: List[str]
    opportunity_type: str
    description: str
    potential_impact: str  # 'high', 'medium', 'low'
    confidence: float
    next_steps: List[str] = field(default_factory=list)


class CollaborationFinder:
    """Find collaboration opportunities in a CE ecosystem.

    This class provides methods to:
    - Find entities by capability
    - Find entities by need
    - Find entities in the same cluster
    - Find complementary partners (capability-need matches)
    - Find value chain partners
    - Find geographic collaborators
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the CollaborationFinder.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def find_by_capability(
        self,
        capability: str,
        entities: List[EntityProfileDict],
        min_similarity: float = 0.7
    ) -> List[CollaborationMatch]:
        """Find entities that offer a specific capability.

        Args:
            capability: The capability to search for
            entities: List of entity profiles
            min_similarity: Minimum text similarity threshold

        Returns:
            List of matching entities with relevance scores
        """
        matches = []
        capability_lower = capability.lower()

        for entity in entities:
            capabilities = entity.get('ce_capabilities_offered', [])
            if not capabilities:
                continue

            matched_caps = []
            max_score = 0.0

            for cap in capabilities:
                cap_name = cap.get('capability_name', '').lower()
                cap_desc = cap.get('description', '').lower()

                # Check for exact or partial match
                if capability_lower in cap_name or capability_lower in cap_desc:
                    score = 1.0 if capability_lower == cap_name else 0.85
                    matched_caps.append(cap.get('capability_name', ''))
                    max_score = max(max_score, score)
                else:
                    # Simple word overlap similarity
                    cap_words = set(cap_name.split() + cap_desc.split())
                    search_words = set(capability_lower.split())
                    if cap_words and search_words:
                        overlap = len(cap_words & search_words)
                        total = len(cap_words | search_words)
                        score = overlap / total if total > 0 else 0
                        if score >= min_similarity:
                            matched_caps.append(cap.get('capability_name', ''))
                            max_score = max(max_score, score)

            if matched_caps:
                matches.append(CollaborationMatch(
                    entity_name=entity.get('entity_name', 'Unknown'),
                    entity_url=str(entity.get('url', '')),
                    match_type='capability',
                    match_reason=f"Offers capability related to '{capability}'",
                    relevance_score=max_score,
                    matched_items=matched_caps
                ))

        # Sort by relevance
        matches.sort(key=lambda x: x.relevance_score, reverse=True)
        self.logger.info(f"Found {len(matches)} entities with capability '{capability}'")
        return matches

    def find_by_need(
        self,
        need: str,
        entities: List[EntityProfileDict],
        min_similarity: float = 0.7
    ) -> List[CollaborationMatch]:
        """Find entities that have a specific need.

        Args:
            need: The need to search for
            entities: List of entity profiles
            min_similarity: Minimum text similarity threshold

        Returns:
            List of matching entities with relevance scores
        """
        matches = []
        need_lower = need.lower()

        for entity in entities:
            needs = entity.get('ce_needs_requirements', [])
            if not needs:
                continue

            matched_needs = []
            max_score = 0.0

            for n in needs:
                need_name = n.get('need_name', '').lower()
                need_desc = n.get('description', '').lower()

                if need_lower in need_name or need_lower in need_desc:
                    score = 1.0 if need_lower == need_name else 0.85
                    matched_needs.append(n.get('need_name', ''))
                    max_score = max(max_score, score)
                else:
                    # Word overlap similarity
                    need_words = set(need_name.split() + need_desc.split())
                    search_words = set(need_lower.split())
                    if need_words and search_words:
                        overlap = len(need_words & search_words)
                        total = len(need_words | search_words)
                        score = overlap / total if total > 0 else 0
                        if score >= min_similarity:
                            matched_needs.append(n.get('need_name', ''))
                            max_score = max(max_score, score)

            if matched_needs:
                matches.append(CollaborationMatch(
                    entity_name=entity.get('entity_name', 'Unknown'),
                    entity_url=str(entity.get('url', '')),
                    match_type='need',
                    match_reason=f"Has need related to '{need}'",
                    relevance_score=max_score,
                    matched_items=matched_needs
                ))

        matches.sort(key=lambda x: x.relevance_score, reverse=True)
        self.logger.info(f"Found {len(matches)} entities with need '{need}'")
        return matches

    def find_by_cluster(
        self,
        cluster_id: str,
        clusters: List[ClusterDict]
    ) -> List[str]:
        """Find all entities in a specific cluster.

        Args:
            cluster_id: The cluster ID to search for
            clusters: List of cluster dictionaries

        Returns:
            List of entity names in the cluster
        """
        for cluster in clusters:
            if cluster.get('cluster_id') == cluster_id:
                entities = cluster.get('entities', [])
                self.logger.info(f"Found {len(entities)} entities in cluster '{cluster_id}'")
                return entities

        self.logger.warning(f"Cluster '{cluster_id}' not found")
        return []

    def find_complementary_partners(
        self,
        entity: EntityProfileDict,
        all_entities: List[EntityProfileDict],
        min_score: float = 0.6
    ) -> List[CollaborationMatch]:
        """Find entities with complementary capabilities/needs.

        Looks for entities that:
        - Have capabilities matching this entity's needs
        - Have needs matching this entity's capabilities

        Args:
            entity: The entity to find partners for
            all_entities: All entities in the ecosystem
            min_score: Minimum relevance score

        Returns:
            List of complementary partner matches
        """
        entity_name = entity.get('entity_name', '')
        entity_capabilities = entity.get('ce_capabilities_offered', [])
        entity_needs = entity.get('ce_needs_requirements', [])

        matches = []

        for other in all_entities:
            other_name = other.get('entity_name', '')
            if other_name == entity_name:
                continue

            other_capabilities = other.get('ce_capabilities_offered', [])
            other_needs = other.get('ce_needs_requirements', [])

            matched_items = []
            total_score = 0.0
            match_count = 0

            # Check if other's capabilities match our needs
            for need in entity_needs:
                need_name = need.get('need_name', '').lower()
                for cap in other_capabilities:
                    cap_name = cap.get('capability_name', '').lower()
                    # Simple keyword overlap
                    need_words = set(need_name.split())
                    cap_words = set(cap_name.split())
                    if need_words & cap_words:
                        matched_items.append(f"Their '{cap.get('capability_name')}' matches your need '{need.get('need_name')}'")
                        total_score += 0.8
                        match_count += 1

            # Check if our capabilities match their needs
            for cap in entity_capabilities:
                cap_name = cap.get('capability_name', '').lower()
                for need in other_needs:
                    need_name = need.get('need_name', '').lower()
                    cap_words = set(cap_name.split())
                    need_words = set(need_name.split())
                    if cap_words & need_words:
                        matched_items.append(f"Your '{cap.get('capability_name')}' matches their need '{need.get('need_name')}'")
                        total_score += 0.8
                        match_count += 1

            if match_count > 0:
                avg_score = total_score / match_count
                if avg_score >= min_score:
                    matches.append(CollaborationMatch(
                        entity_name=other_name,
                        entity_url=str(other.get('url', '')),
                        match_type='complementary',
                        match_reason=f"Complementary capabilities/needs found",
                        relevance_score=min(avg_score, 1.0),
                        matched_items=matched_items
                    ))

        matches.sort(key=lambda x: x.relevance_score, reverse=True)
        self.logger.info(f"Found {len(matches)} complementary partners for '{entity_name}'")
        return matches

    def find_value_chain_partners(
        self,
        entity: EntityProfileDict,
        all_entities: List[EntityProfileDict]
    ) -> List[CollaborationMatch]:
        """Find upstream/downstream value chain partners.

        Identifies entities that could be:
        - Suppliers (upstream)
        - Customers/processors (downstream)
        - Parallel collaborators (same stage)

        Args:
            entity: The entity to find partners for
            all_entities: All entities in the ecosystem

        Returns:
            List of value chain partner matches
        """
        # Define value chain stages
        value_chain_stages = {
            'design': ['design', 'eco-design', 'circular design', 'product development'],
            'production': ['manufacturing', 'production', 'fabrication', 'assembly'],
            'distribution': ['logistics', 'distribution', 'transport', 'delivery'],
            'use': ['retail', 'rental', 'leasing', 'service', 'maintenance', 'repair'],
            'collection': ['collection', 'take-back', 'return', 'gathering'],
            'recycling': ['recycling', 'processing', 'recovery', 'remanufacturing'],
            'materials': ['raw material', 'secondary material', 'recycled material']
        }

        entity_name = entity.get('entity_name', '')
        entity_activities = entity.get('ce_activities_structured', [])

        # Determine entity's stage(s)
        entity_stages = set()
        for activity in entity_activities:
            activity_name = activity.get('activity_name', '').lower()
            for stage, keywords in value_chain_stages.items():
                if any(kw in activity_name for kw in keywords):
                    entity_stages.add(stage)

        if not entity_stages:
            self.logger.debug(f"Could not determine value chain stage for '{entity_name}'")
            return []

        # Define stage relationships
        stage_order = ['design', 'production', 'distribution', 'use', 'collection', 'recycling', 'materials']

        matches = []

        for other in all_entities:
            other_name = other.get('entity_name', '')
            if other_name == entity_name:
                continue

            other_activities = other.get('ce_activities_structured', [])
            other_stages = set()

            for activity in other_activities:
                activity_name = activity.get('activity_name', '').lower()
                for stage, keywords in value_chain_stages.items():
                    if any(kw in activity_name for kw in keywords):
                        other_stages.add(stage)

            if not other_stages:
                continue

            # Check relationship
            relationship = None
            matched_items = []

            for entity_stage in entity_stages:
                try:
                    entity_idx = stage_order.index(entity_stage)
                except ValueError:
                    continue

                for other_stage in other_stages:
                    try:
                        other_idx = stage_order.index(other_stage)
                    except ValueError:
                        continue

                    if other_idx == entity_idx - 1:
                        relationship = 'upstream_supplier'
                        matched_items.append(f"Upstream: {other_stage} -> {entity_stage}")
                    elif other_idx == entity_idx + 1:
                        relationship = 'downstream_processor'
                        matched_items.append(f"Downstream: {entity_stage} -> {other_stage}")
                    elif other_idx == entity_idx and entity_stage == other_stage:
                        relationship = 'parallel_collaborator'
                        matched_items.append(f"Same stage: {entity_stage}")

            if relationship and matched_items:
                matches.append(CollaborationMatch(
                    entity_name=other_name,
                    entity_url=str(other.get('url', '')),
                    match_type='value_chain',
                    match_reason=f"Value chain partner ({relationship.replace('_', ' ')})",
                    relevance_score=0.75,
                    matched_items=matched_items
                ))

        self.logger.info(f"Found {len(matches)} value chain partners for '{entity_name}'")
        return matches

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers."""
        R = 6371  # Earth's radius in km

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c

    def find_geographic_collaborators(
        self,
        entity: EntityProfileDict,
        all_entities: List[EntityProfileDict],
        radius_km: float = 10.0
    ) -> List[CollaborationMatch]:
        """Find nearby entities for geographic collaboration.

        Args:
            entity: The entity to find collaborators for
            all_entities: All entities in the ecosystem
            radius_km: Maximum distance in kilometers

        Returns:
            List of nearby entity matches
        """
        entity_name = entity.get('entity_name', '')
        entity_address = entity.get('address', '')

        # Note: This is a simplified implementation
        # A full implementation would use geocoding
        # For now, we match by address similarity (same city/district)

        matches = []

        if not entity_address:
            return matches

        entity_addr_lower = entity_address.lower()
        entity_addr_words = set(entity_addr_lower.split())

        for other in all_entities:
            other_name = other.get('entity_name', '')
            if other_name == entity_name:
                continue

            other_address = other.get('address', '')
            if not other_address:
                continue

            other_addr_lower = other_address.lower()
            other_addr_words = set(other_addr_lower.split())

            # Calculate address word overlap
            overlap = entity_addr_words & other_addr_words
            if len(overlap) >= 2:  # At least 2 words in common (e.g., city + street type)
                relevance = len(overlap) / max(len(entity_addr_words), len(other_addr_words))

                if relevance >= 0.3:  # 30% overlap threshold
                    matches.append(CollaborationMatch(
                        entity_name=other_name,
                        entity_url=str(other.get('url', '')),
                        match_type='geographic',
                        match_reason=f"Located in similar area",
                        relevance_score=relevance,
                        matched_items=[f"Address overlap: {', '.join(list(overlap)[:5])}"]
                    ))

        matches.sort(key=lambda x: x.relevance_score, reverse=True)
        self.logger.info(f"Found {len(matches)} geographic collaborators for '{entity_name}'")
        return matches

    def find_all_collaboration_opportunities(
        self,
        entity: EntityProfileDict,
        all_entities: List[EntityProfileDict],
        clusters: List[ClusterDict] = None
    ) -> List[CollaborationOpportunity]:
        """Find all collaboration opportunities for an entity.

        Combines all matching methods to provide comprehensive opportunities.

        Args:
            entity: The entity to find opportunities for
            all_entities: All entities in the ecosystem
            clusters: Optional list of clusters

        Returns:
            List of collaboration opportunities sorted by potential impact
        """
        entity_name = entity.get('entity_name', '')
        opportunities = []

        # Find complementary partners
        complementary = self.find_complementary_partners(entity, all_entities)
        for match in complementary[:5]:  # Top 5
            opportunities.append(CollaborationOpportunity(
                entities=[entity_name, match.entity_name],
                opportunity_type='capability_need_match',
                description=f"{entity_name} and {match.entity_name} have complementary capabilities and needs",
                potential_impact='high' if match.relevance_score >= 0.8 else 'medium',
                confidence=match.relevance_score,
                next_steps=[
                    f"Contact {match.entity_name} to discuss collaboration",
                    "Share capability profiles",
                    "Identify specific project opportunities"
                ]
            ))

        # Find value chain partners
        value_chain = self.find_value_chain_partners(entity, all_entities)
        for match in value_chain[:3]:  # Top 3
            opportunities.append(CollaborationOpportunity(
                entities=[entity_name, match.entity_name],
                opportunity_type='value_chain_partnership',
                description=f"{entity_name} and {match.entity_name} could collaborate in the circular value chain",
                potential_impact='high',
                confidence=match.relevance_score,
                next_steps=[
                    "Map material/product flows between entities",
                    "Identify synergies in logistics",
                    "Explore joint circular business models"
                ]
            ))

        # Find geographic collaborators
        geographic = self.find_geographic_collaborators(entity, all_entities)
        for match in geographic[:3]:  # Top 3
            opportunities.append(CollaborationOpportunity(
                entities=[entity_name, match.entity_name],
                opportunity_type='geographic_cluster',
                description=f"{entity_name} and {match.entity_name} are located in the same area",
                potential_impact='medium',
                confidence=match.relevance_score,
                next_steps=[
                    "Explore shared infrastructure opportunities",
                    "Consider joint waste collection/processing",
                    "Investigate co-location benefits"
                ]
            ))

        # Sort by impact and confidence
        impact_order = {'high': 3, 'medium': 2, 'low': 1}
        opportunities.sort(
            key=lambda x: (impact_order.get(x.potential_impact, 0), x.confidence),
            reverse=True
        )

        self.logger.info(f"Found {len(opportunities)} collaboration opportunities for '{entity_name}'")
        return opportunities


__all__ = [
    'CollaborationFinder',
    'CollaborationMatch',
    'CollaborationOpportunity',
]
