#!/usr/bin/env python3
"""
Apply Stage 4 Batched Clustering Fixes to All Ecosystems

This script applies the Stage 4 token overflow fixes to all ecosystem configurations:
1. Helper methods for batching and cluster merging
2. Refactored clustering methods with batching support
3. Updated configuration settings

Fixes are applied to:
- novi_sad_ce_ecosystem
- cahul_ce_ecosystem
- CE-Ecosystem Builder (vLLM + Guidance)/hamburg_ce_ecosystem
- CE-Ecosystem Builder (vLLM + Guidance)/novi_sad_ce_ecosystem
- CE-Ecosystem Builder (vLLM + Guidance)/cahul_ce_ecosystem
"""

import os
import re
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
VLLM_BASE_DIR = BASE_DIR.parent / "CE-Ecosystem Builder (vLLM + Guidance)"

# Source files (Hamburg as reference)
HAMBURG_RELATIONSHIP_ANALYZER = BASE_DIR / "hamburg_ce_ecosystem" / "scrapers" / "relationship_analyzer.py"
HAMBURG_CONFIG = BASE_DIR / "hamburg_ce_ecosystem" / "config" / "scrape_config.yaml"

# Target ecosystems (excluding Hamburg which already has the fixes)
ECOSYSTEMS = [
    ("novi_sad_ce_ecosystem", BASE_DIR),
    ("cahul_ce_ecosystem", BASE_DIR),
    ("hamburg_ce_ecosystem", VLLM_BASE_DIR),
    ("novi_sad_ce_ecosystem", VLLM_BASE_DIR),
    ("cahul_ce_ecosystem", VLLM_BASE_DIR),
]


def read_file(filepath: Path) -> str:
    """Read file contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(filepath: Path, content: str):
    """Write content to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def extract_helper_methods(content: str) -> str:
    """Extract the three helper methods from Hamburg's relationship_analyzer.py."""
    # Find _batch_items method
    batch_items_start = content.find("    def _batch_items(self, items: List[Any], batch_size: int)")
    if batch_items_start == -1:
        raise ValueError("Could not find _batch_items method")

    # Find _merge_clusters end (next method after it)
    merge_end = content.find("\n    def cluster_ce_capabilities(", batch_items_start)
    if merge_end == -1:
        raise ValueError("Could not find end of helper methods")

    return content[batch_items_start:merge_end]


def extract_clustering_method(content: str, method_name: str) -> str:
    """Extract a specific clustering method."""
    # Find method start - try different indentation levels
    method_start = -1
    for indent in ["    ", "  ", "\t"]:
        method_pattern = f"{indent}def {method_name}("
        method_start = content.find(method_pattern)
        if method_start != -1:
            break

    if method_start == -1:
        raise ValueError(f"Could not find {method_name} method")

    # Find next method (end of current method) - use same indentation
    next_method_start = -1
    for indent in ["    ", "  ", "\t"]:
        pattern = f"\n{indent}def "
        pos = content.find(pattern, method_start + 10)
        if pos != -1:
            next_method_start = pos
            break

    if next_method_start == -1:
        # This is the last method, find class end or file end
        next_method_start = len(content)

    return content[method_start:next_method_start]


def apply_relationship_analyzer_fixes(ecosystem_name: str, base_path: Path):
    """Apply fixes to relationship_analyzer.py for a specific ecosystem."""
    target_file = base_path / ecosystem_name / "scrapers" / "relationship_analyzer.py"

    if not target_file.exists():
        print(f"  ⚠️  relationship_analyzer.py not found at {target_file}")
        return False

    print(f"  📝 Applying fixes to relationship_analyzer.py...")

    # Read source and target
    source_content = read_file(HAMBURG_RELATIONSHIP_ANALYZER)
    target_content = read_file(target_file)

    # Extract components from Hamburg
    helper_methods = extract_helper_methods(source_content)
    cluster_capabilities = extract_clustering_method(source_content, "cluster_ce_capabilities")
    cluster_needs = extract_clustering_method(source_content, "cluster_ce_needs")
    cluster_activities = extract_clustering_method(source_content, "cluster_ce_activities")

    # Replace import references if needed (hamburg -> novi_sad, etc.)
    if ecosystem_name != "hamburg_ce_ecosystem":
        helper_methods = helper_methods.replace("hamburg_ce_ecosystem", ecosystem_name)
        cluster_capabilities = cluster_capabilities.replace("hamburg_ce_ecosystem", ecosystem_name)
        cluster_needs = cluster_needs.replace("hamburg_ce_ecosystem", ecosystem_name)
        cluster_activities = cluster_activities.replace("hamburg_ce_ecosystem", ecosystem_name)

    # Check if helper methods already exist
    if "_batch_items" in target_content:
        print("    ℹ️  Helper methods already exist, replacing...")
        # Remove old helper methods - try different indentations
        for indent in ["    ", "  ", "\t"]:
            batch_start = target_content.find(f"{indent}def _batch_items(")
            if batch_start != -1:
                merge_end = target_content.find(f"\n{indent}def cluster_ce_capabilities(", batch_start)
                if merge_end != -1:
                    target_content = target_content[:batch_start] + target_content[merge_end:]
                    break

    # Insert helper methods before cluster_ce_capabilities - try different indentations
    capabilities_pos = -1
    for indent in ["    ", "  ", "\t"]:
        pos = target_content.find(f"{indent}def cluster_ce_capabilities(")
        if pos != -1:
            capabilities_pos = pos
            break

    if capabilities_pos == -1:
        print("    ❌ Could not find cluster_ce_capabilities method")
        return False

    target_content = (
        target_content[:capabilities_pos] +
        helper_methods + "\n" +
        target_content[capabilities_pos:]
    )

    # Replace cluster_ce_capabilities
    old_capabilities_start = -1
    old_capabilities_end = -1
    for indent in ["    ", "  ", "\t"]:
        start = target_content.find(f"{indent}def cluster_ce_capabilities(")
        if start != -1:
            old_capabilities_start = start
            end = target_content.find(f"\n{indent}def cluster_ce_needs(", start)
            if end != -1:
                old_capabilities_end = end
            break

    if old_capabilities_start != -1 and old_capabilities_end != -1:
        target_content = (
            target_content[:old_capabilities_start] +
            cluster_capabilities +
            target_content[old_capabilities_end:]
        )

    # Replace cluster_ce_needs
    old_needs_start = -1
    old_needs_end = -1
    for indent in ["    ", "  ", "\t"]:
        start = target_content.find(f"{indent}def cluster_ce_needs(")
        if start != -1:
            old_needs_start = start
            end = target_content.find(f"\n{indent}def cluster_ce_activities(", start)
            if end != -1:
                old_needs_end = end
            break

    if old_needs_start != -1 and old_needs_end != -1:
        target_content = (
            target_content[:old_needs_start] +
            cluster_needs +
            target_content[old_needs_end:]
        )

    # Replace cluster_ce_activities
    old_activities_start = -1
    old_activities_end = -1
    for indent in ["    ", "  ", "\t"]:
        start = target_content.find(f"{indent}def cluster_ce_activities(")
        if start != -1:
            old_activities_start = start
            # Find next method
            end = target_content.find(f"\n{indent}def ", start + 10)
            if end == -1:
                # Last method in class
                end = len(target_content)
            old_activities_end = end
            break

    if old_activities_start != -1:
        target_content = (
            target_content[:old_activities_start] +
            cluster_activities +
            target_content[old_activities_end:]
        )

    # Write back
    write_file(target_file, target_content)
    print("    ✅ relationship_analyzer.py updated successfully")
    return True


def apply_config_fixes(ecosystem_name: str, base_path: Path):
    """Apply clustering configuration fixes to scrape_config.yaml."""
    target_file = base_path / ecosystem_name / "config" / "scrape_config.yaml"

    if not target_file.exists():
        print(f"  ⚠️  scrape_config.yaml not found at {target_file}")
        return False

    print(f"  📝 Checking scrape_config.yaml...")

    target_content = read_file(target_file)

    # Check if clustering config already has batch_size
    if "batch_size: 60" in target_content:
        print("    ℹ️  Clustering config already updated")
        return True

    # Find clustering section
    clustering_start = target_content.find("  clustering:")
    if clustering_start == -1:
        print("    ⚠️  No clustering section found")
        return False

    # Check if we need to add batch_size, max_prompt_tokens, similarity_threshold
    clustering_section_end = target_content.find("\n  # ", clustering_start + 10)
    if clustering_section_end == -1:
        clustering_section_end = target_content.find("\n\n", clustering_start + 10)

    clustering_section = target_content[clustering_start:clustering_section_end]

    if "batch_size:" not in clustering_section:
        # Add the new config keys after activity_clusters
        activity_line_pos = target_content.find("activity_clusters:", clustering_start)
        if activity_line_pos != -1:
            # Find end of line
            line_end = target_content.find("\n", activity_line_pos)
            insert_text = (
                "\n    batch_size: 60  # Max items per batch (fits in 4096 token window, ~3500 tokens for data)"
                "\n    max_prompt_tokens: 3500  # Safety limit for 4096 token context window"
                "\n    similarity_threshold: 0.80  # Threshold for merging similar clusters (0.0-1.0)"
            )
            target_content = target_content[:line_end] + insert_text + target_content[line_end:]
            write_file(target_file, target_content)
            print("    ✅ scrape_config.yaml updated successfully")
            return True

    print("    ✅ scrape_config.yaml already has correct config")
    return True


def main():
    print("=" * 80)
    print("Stage 4 Batched Clustering Fixes - Apply to All Ecosystems")
    print("=" * 80)
    print()

    # Verify source files exist
    if not HAMBURG_RELATIONSHIP_ANALYZER.exists():
        print(f"❌ Source file not found: {HAMBURG_RELATIONSHIP_ANALYZER}")
        return

    if not HAMBURG_CONFIG.exists():
        print(f"❌ Source file not found: {HAMBURG_CONFIG}")
        return

    print(f"✅ Source files found")
    print(f"   - {HAMBURG_RELATIONSHIP_ANALYZER}")
    print(f"   - {HAMBURG_CONFIG}")
    print()

    # Apply fixes to each ecosystem
    success_count = 0
    total_count = len(ECOSYSTEMS)

    for ecosystem_name, base_path in ECOSYSTEMS:
        print(f"📦 Processing: {ecosystem_name} ({base_path.name})")

        # Check if ecosystem directory exists
        ecosystem_path = base_path / ecosystem_name
        if not ecosystem_path.exists():
            print(f"  ⚠️  Ecosystem directory not found: {ecosystem_path}")
            print()
            continue

        try:
            # Apply fixes
            analyzer_ok = apply_relationship_analyzer_fixes(ecosystem_name, base_path)
            config_ok = apply_config_fixes(ecosystem_name, base_path)

            if analyzer_ok and config_ok:
                print(f"  ✅ {ecosystem_name} updated successfully")
                success_count += 1
            else:
                print(f"  ⚠️  {ecosystem_name} partially updated")

        except Exception as e:
            print(f"  ❌ Error updating {ecosystem_name}: {e}")

        print()

    # Summary
    print("=" * 80)
    print(f"Summary: {success_count}/{total_count} ecosystems updated successfully")
    print("=" * 80)
    print()

    if success_count == total_count:
        print("✅ All Stage 4 fixes applied successfully!")
        print()
        print("Next steps:")
        print("1. Test with Hamburg ecosystem: cd 'CE-Ecosystem Builder' && python3 run_stages_3_and_4.py")
        print("2. Monitor Ollama logs for truncation warnings")
        print("3. Validate cluster distribution and quality")
    else:
        print("⚠️  Some ecosystems could not be updated. Please review the errors above.")


if __name__ == "__main__":
    main()
