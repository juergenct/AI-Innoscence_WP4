# circular_scraper/utils/export_manager.py
"""
Data export and consolidation utilities
Handles merging and exporting scraped data to various formats
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class ExportManager:
    """
    Manages data export and consolidation
    Merges multiple scraping sessions and exports to different formats
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize export manager
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.export_dir = self.data_dir / 'exports'
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.entities_df = None
        self.links_df = None
        self.errors_df = None
        
        logger.info(f"ExportManager initialized with data directory: {self.data_dir}")
    
    def consolidate_session_data(self, session_id: str = None) -> Dict[str, pd.DataFrame]:
        """
        Consolidate data from a scraping session
        
        Args:
            session_id: Specific session to consolidate (or latest if None)
        
        Returns:
            Dictionary with consolidated DataFrames
        """
        csv_dir = self.export_dir / 'csv'
        
        if not csv_dir.exists():
            logger.error(f"CSV directory not found: {csv_dir}")
            return {}
        
        # Find session files
        if session_id:
            entity_files = list(csv_dir.glob(f"entities_{session_id}_*.csv"))
            link_files = list(csv_dir.glob(f"links_{session_id}_*.csv"))
            error_files = list(csv_dir.glob(f"errors_{session_id}_*.csv"))
        else:
            # Get all files
            entity_files = list(csv_dir.glob("entities_*.csv"))
            link_files = list(csv_dir.glob("links_*.csv"))
            error_files = list(csv_dir.glob("errors_*.csv"))
        
        # Consolidate entities
        if entity_files:
            entity_dfs = []
            for file in entity_files:
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                    entity_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")
            
            if entity_dfs:
                self.entities_df = pd.concat(entity_dfs, ignore_index=True)
                self._clean_entities_df()
                logger.info(f"Consolidated {len(self.entities_df)} entities from {len(entity_files)} files")
        
        # Consolidate links
        if link_files:
            link_dfs = []
            for file in link_files:
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                    link_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")
            
            if link_dfs:
                self.links_df = pd.concat(link_dfs, ignore_index=True)
                logger.info(f"Consolidated {len(self.links_df)} links from {len(link_files)} files")
        
        # Consolidate errors
        if error_files:
            error_dfs = []
            for file in error_files:
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                    error_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")
            
            if error_dfs:
                self.errors_df = pd.concat(error_dfs, ignore_index=True)
                logger.info(f"Consolidated {len(self.errors_df)} errors from {len(error_files)} files")
        
        return {
            'entities': self.entities_df,
            'links': self.links_df,
            'errors': self.errors_df
        }
    
    def _clean_entities_df(self):
        """Clean and deduplicate entities DataFrame"""
        if self.entities_df is None or self.entities_df.empty:
            return
        
        # Remove duplicates based on URL
        self.entities_df = self.entities_df.drop_duplicates(subset=['url'], keep='last')
        
        # Convert boolean columns
        bool_columns = ['has_circular_economy_terms', 'has_hamburg_reference']
        for col in bool_columns:
            if col in self.entities_df.columns:
                self.entities_df[col] = self.entities_df[col].fillna(False).astype(bool)
        
        # Clean text fields
        text_columns = ['title', 'organization_name', 'city']
        for col in text_columns:
            if col in self.entities_df.columns:
                self.entities_df[col] = self.entities_df[col].fillna('').str.strip()
        
        # Sort by relevance
        self.entities_df['relevance_score'] = (
            self.entities_df.get('has_circular_economy_terms', False).astype(int) * 2 +
            self.entities_df.get('has_hamburg_reference', False).astype(int)
        )
        
        self.entities_df = self.entities_df.sort_values('relevance_score', ascending=False)
    
    def export_to_excel(self, output_file: str = None) -> str:
        """
        Export consolidated data to Excel with multiple sheets
        
        Args:
            output_file: Output file path
        
        Returns:
            Path to exported file
        """
        if output_file is None:
            output_file = self.export_dir / f"circular_economy_data_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        
        output_file = Path(output_file)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main entities sheet
            if self.entities_df is not None and not self.entities_df.empty:
                # Select important columns
                export_cols = [
                    'url', 'domain', 'entity_id', 'entity_name', 'entity_root_url',
                    'title', 'organization_name', 'city',
                    'has_circular_economy_terms', 'has_hamburg_reference',
                    'language', 'emails', 'phone_numbers', 'scraped_at'
                ]
                
                export_cols = [col for col in export_cols if col in self.entities_df.columns]
                entities_export = self.entities_df[export_cols].copy()
                
                entities_export.to_excel(writer, sheet_name='Entities', index=False)
                
                # Format the Excel sheet
                worksheet = writer.sheets['Entities']
                for column in entities_export:
                    column_width = max(
                        entities_export[column].astype(str).map(len).max(),
                        len(column)
                    ) + 2
                    column_width = min(column_width, 50)  # Max width
                    col_idx = entities_export.columns.get_loc(column)
                    worksheet.column_dimensions[chr(65 + col_idx)].width = column_width
            
            # Relevant entities sheet (filtered)
            if self.entities_df is not None and not self.entities_df.empty:
                relevant = self.entities_df[
                    (self.entities_df.get('has_circular_economy_terms', False)) &
                    (self.entities_df.get('has_hamburg_reference', False))
                ]
                
                if not relevant.empty:
                    relevant[export_cols].to_excel(
                        writer, 
                        sheet_name='Relevant_Entities', 
                        index=False
                    )
            
            # Statistics sheet
            stats_df = self._generate_statistics()
            if stats_df is not None:
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Links sheet (if not too large)
            if self.links_df is not None and len(self.links_df) < 10000:
                self.links_df.head(5000).to_excel(
                    writer, 
                    sheet_name='Links_Sample', 
                    index=False
                )
            
            # Errors sheet
            if self.errors_df is not None and not self.errors_df.empty:
                self.errors_df.to_excel(writer, sheet_name='Errors', index=False)
        
        logger.info(f"Exported data to Excel: {output_file}")
        return str(output_file)

    def export_entities_table(self) -> pd.DataFrame:
        """Aggregate entities with counts for quick review."""
        if self.entities_df is None or self.entities_df.empty:
            return pd.DataFrame()
        cols = ['entity_id', 'entity_name', 'entity_root_url', 'domain']
        for c in cols:
            if c not in self.entities_df.columns:
                self.entities_df[c] = ''
        agg = (self.entities_df
               .groupby(['entity_id', 'entity_name', 'entity_root_url', 'domain'], dropna=False)
               .agg(pages=('url', 'count'),
                    ce_pages=('has_circular_economy_terms', 'sum'),
                    hh_pages=('has_hamburg_reference', 'sum'))
               .reset_index()
               .sort_values('pages', ascending=False))
        out = self.export_dir / 'entities_aggregated.csv'
        agg.to_csv(out, index=False)
        logger.info(f"Exported aggregated entities table to {out}")
        return agg
    
    def export_to_parquet(self, output_dir: str = None) -> Dict[str, str]:
        """
        Export data to Parquet format for efficient storage
        
        Args:
            output_dir: Output directory
        
        Returns:
            Dictionary with paths to exported files
        """
        if output_dir is None:
            output_dir = self.export_dir / 'parquet' / f"{datetime.now():%Y%m%d_%H%M%S}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export entities
        if self.entities_df is not None and not self.entities_df.empty:
            entities_file = output_dir / 'entities.parquet'
            self.entities_df.to_parquet(entities_file, engine='pyarrow', compression='snappy')
            exported_files['entities'] = str(entities_file)
            logger.info(f"Exported {len(self.entities_df)} entities to {entities_file}")
        
        # Export links
        if self.links_df is not None and not self.links_df.empty:
            links_file = output_dir / 'links.parquet'
            self.links_df.to_parquet(links_file, engine='pyarrow', compression='snappy')
            exported_files['links'] = str(links_file)
            logger.info(f"Exported {len(self.links_df)} links to {links_file}")
        
        # Export errors
        if self.errors_df is not None and not self.errors_df.empty:
            errors_file = output_dir / 'errors.parquet'
            self.errors_df.to_parquet(errors_file, engine='pyarrow', compression='snappy')
            exported_files['errors'] = str(errors_file)
            logger.info(f"Exported {len(self.errors_df)} errors to {errors_file}")
        
        return exported_files
    
    def export_for_llm_analysis(self, output_file: str = None) -> str:
        """
        Export data in format optimized for LLM analysis
        
        Args:
            output_file: Output file path
        
        Returns:
            Path to exported file
        """
        if output_file is None:
            output_file = self.export_dir / f"llm_analysis_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        
        output_file = Path(output_file)
        
        if self.entities_df is None or self.entities_df.empty:
            logger.warning("No entities to export for LLM analysis")
            return None
        
        # Prepare data for LLM
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in self.entities_df.iterrows():
                # Create structured record for LLM analysis
                record = {
                    'url': row.get('url', ''),
                    'domain': row.get('domain', ''),
                    'title': row.get('title', ''),
                    'organization_name': row.get('organization_name', ''),
                    'location': {
                        'city': row.get('city', ''),
                        'has_hamburg_reference': bool(row.get('has_hamburg_reference', False))
                    },
                    'content_indicators': {
                        'has_circular_economy_terms': bool(row.get('has_circular_economy_terms', False)),
                        'language': row.get('language', ''),
                        'content_length': int(row.get('content_length', 0))
                    },
                    'contacts': {
                        'emails': row.get('emails', '').split(';') if row.get('emails') else [],
                        'phones': row.get('phone_numbers', '').split(';') if row.get('phone_numbers') else []
                    },
                    'links_count': {
                        'internal': int(row.get('internal_links_count', 0)),
                        'external': int(row.get('external_links_count', 0))
                    },
                    'metadata': {
                        'scraped_at': row.get('scraped_at', ''),
                        'crawl_depth': int(row.get('crawl_depth', 0))
                    }
                }
                
                # Write as JSON line
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Exported {len(self.entities_df)} records for LLM analysis to {output_file}")
        return str(output_file)
    
    def _generate_statistics(self) -> pd.DataFrame:
        """
        Generate statistics DataFrame
        
        Returns:
            DataFrame with statistics
        """
        stats = []
        
        if self.entities_df is not None and not self.entities_df.empty:
            # Overall statistics
            stats.append({
                'Metric': 'Total Entities',
                'Value': len(self.entities_df)
            })
            
            stats.append({
                'Metric': 'Unique Domains',
                'Value': self.entities_df['domain'].nunique()
            })
            
            # Relevance statistics
            ce_count = self.entities_df['has_circular_economy_terms'].sum() if 'has_circular_economy_terms' in self.entities_df else 0
            hh_count = self.entities_df['has_hamburg_reference'].sum() if 'has_hamburg_reference' in self.entities_df else 0
            both_count = len(self.entities_df[
                (self.entities_df.get('has_circular_economy_terms', False)) &
                (self.entities_df.get('has_hamburg_reference', False))
            ])
            
            stats.extend([
                {'Metric': 'Circular Economy Related', 'Value': ce_count},
                {'Metric': 'Hamburg Referenced', 'Value': hh_count},
                {'Metric': 'Both CE and Hamburg', 'Value': both_count},
            ])
            
            # Language distribution
            if 'language' in self.entities_df:
                lang_dist = self.entities_df['language'].value_counts().head(5)
                for lang, count in lang_dist.items():
                    stats.append({
                        'Metric': f'Language: {lang}',
                        'Value': count
                    })
            
            # Contact information
            if 'emails' in self.entities_df:
                has_email = self.entities_df['emails'].notna() & (self.entities_df['emails'] != '')
                stats.append({
                    'Metric': 'Entities with Email',
                    'Value': has_email.sum()
                })
            
            if 'phone_numbers' in self.entities_df:
                has_phone = self.entities_df['phone_numbers'].notna() & (self.entities_df['phone_numbers'] != '')
                stats.append({
                    'Metric': 'Entities with Phone',
                    'Value': has_phone.sum()
                })
        
        if self.links_df is not None and not self.links_df.empty:
            stats.append({
                'Metric': 'Total Links Discovered',
                'Value': len(self.links_df)
            })
        
        if self.errors_df is not None and not self.errors_df.empty:
            stats.append({
                'Metric': 'Total Errors',
                'Value': len(self.errors_df)
            })
            
            if 'error_type' in self.errors_df:
                top_error = self.errors_df['error_type'].value_counts().head(1)
                if not top_error.empty:
                    stats.append({
                        'Metric': f'Most Common Error',
                        'Value': f"{top_error.index[0]} ({top_error.values[0]})"
                    })
        
        return pd.DataFrame(stats) if stats else None
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary report
        
        Returns:
            Summary report as string
        """
        report = []
        report.append("=" * 60)
        report.append("CIRCULAR ECONOMY HAMBURG - SCRAPING SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
        report.append("")
        
        if self.entities_df is not None and not self.entities_df.empty:
            report.append("PAGES SCRAPED")
            report.append("-" * 40)
            report.append(f"Total pages: {len(self.entities_df)}")
            report.append(f"Unique domains: {self.entities_df['domain'].nunique()}")
            # Approximate unique entities by entity_id if present
            if 'entity_id' in self.entities_df.columns:
                report.append(f"Unique entities: {self.entities_df['entity_id'].nunique()}")
            
            if 'has_circular_economy_terms' in self.entities_df:
                ce_count = self.entities_df['has_circular_economy_terms'].sum()
                report.append(f"Circular Economy related: {ce_count}")
            
            if 'has_hamburg_reference' in self.entities_df:
                hh_count = self.entities_df['has_hamburg_reference'].sum()
                report.append(f"Hamburg referenced: {hh_count}")
            
            # Top domains
            report.append("")
            report.append("TOP DOMAINS")
            report.append("-" * 40)
            top_domains = self.entities_df['domain'].value_counts().head(10)
            for domain, count in top_domains.items():
                report.append(f"  {domain}: {count} pages")
            
            # Relevant entities
            relevant = self.entities_df[
                (self.entities_df.get('has_circular_economy_terms', False)) &
                (self.entities_df.get('has_hamburg_reference', False))
            ]
            
            if not relevant.empty:
                report.append("")
                report.append("HIGHLY RELEVANT ENTITIES (CE + Hamburg)")
                report.append("-" * 40)
                
                for _, entity in relevant.head(20).iterrows():
                    report.append(f"  • {entity.get('organization_name', entity.get('title', 'Unknown'))}")
                    report.append(f"    URL: {entity['url']}")
                    if entity.get('emails'):
                        report.append(f"    Contact: {entity['emails']}")
                    report.append("")
        
        if self.errors_df is not None and not self.errors_df.empty:
            report.append("")
            report.append("ERRORS ENCOUNTERED")
            report.append("-" * 40)
            report.append(f"Total errors: {len(self.errors_df)}")
            
            if 'error_type' in self.errors_df:
                error_types = self.errors_df['error_type'].value_counts().head(5)
                for error_type, count in error_types.items():
                    report.append(f"  {error_type}: {count}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export and consolidate scraped data')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--session', help='Specific session ID to export')
    parser.add_argument('--format', choices=['excel', 'parquet', 'llm', 'all'], 
                       default='excel', help='Export format')
    parser.add_argument('--summary', action='store_true', help='Print summary report')
    
    args = parser.parse_args()
    
    # Initialize export manager
    manager = ExportManager(args.data_dir)
    
    # Consolidate data
    data = manager.consolidate_session_data(args.session)
    
    if not data or all(df is None or df.empty for df in data.values()):
        print("No data found to export")
        return
    
    # Export based on format
    if args.format in ['excel', 'all']:
        excel_file = manager.export_to_excel()
        print(f"Exported to Excel: {excel_file}")
    
    if args.format in ['parquet', 'all']:
        parquet_files = manager.export_to_parquet()
        print(f"Exported to Parquet: {parquet_files}")
    
    if args.format in ['llm', 'all']:
        llm_file = manager.export_for_llm_analysis()
        print(f"Exported for LLM analysis: {llm_file}")
    
    # Print summary if requested
    if args.summary:
        print("\n" + manager.generate_summary_report())


if __name__ == '__main__':
    main()