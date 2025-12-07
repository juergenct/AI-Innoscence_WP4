#!/usr/bin/env python3
"""
Apply geocoding fixes to all CE Ecosystem batch_processor.py files.

This script:
1. Reads the enhanced geocode_entities method from Hamburg
2. Applies it to all other ecosystems (Novi Sad, Cahul, vLLM versions)
3. Adjusts the import statement for each ecosystem
"""

import re
from pathlib import Path


def read_hamburg_geocode_method():
    """Read the enhanced geocode_entities method from Hamburg."""
    hamburg_file = Path("/home/thiesen/Documents/AI-Innoscence_Ecosystem/CE-Ecosystem Builder/hamburg_ce_ecosystem/scrapers/batch_processor.py")
    content = hamburg_file.read_text()

    # Extract the geocode_entities method
    pattern = r'(    def geocode_entities\(self.*?\n        return profiles\n)'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        return match.group(1)
    else:
        raise ValueError("Could not find geocode_entities method")


def replace_geocode_method(file_path: Path, new_method: str, ecosystem_name: str):
    """Replace the geocode_entities method in a batch_processor.py file."""

    # Adjust import statement for the specific ecosystem
    adjusted_method = new_method.replace(
        "from hamburg_ce_ecosystem.utils.google_maps_geocoder import GoogleMapsGeocoder",
        f"from {ecosystem_name}.utils.google_maps_geocoder import GoogleMapsGeocoder"
    )

    content = file_path.read_text()

    # Find and replace the geocode_entities method
    pattern = r'    def geocode_entities\(self.*?\n(?=    def \w+|\Z)'

    # Check if method exists
    if not re.search(pattern, content, re.DOTALL):
        print(f"  ⚠️  Could not find geocode_entities method in {file_path}")
        return False

    # Replace the method
    new_content = re.sub(pattern, adjusted_method + "\n", content, count=1, flags=re.DOTALL)

    # Write back
    file_path.write_text(new_content)
    return True


def main():
    """Main execution."""
    print("="*80)
    print("APPLYING GEOCODING FIXES TO ALL ECOSYSTEMS")
    print("="*80)

    # Read the enhanced method from Hamburg
    print("\n📖 Reading enhanced geocode_entities method from Hamburg...")
    try:
        hamburg_method = read_hamburg_geocode_method()
        print(f"✓ Read method ({len(hamburg_method)} characters)")
    except Exception as e:
        print(f"✗ Failed to read Hamburg method: {e}")
        return

    # Define all ecosystems to update
    ecosystems = [
        ("CE-Ecosystem Builder", "novi_sad_ce_ecosystem"),
        ("CE-Ecosystem Builder", "cahul_ce_ecosystem"),
        ("CE-Ecosystem Builder (vLLM + Guidance)", "hamburg_ce_ecosystem"),
        ("CE-Ecosystem Builder (vLLM + Guidance)", "novi_sad_ce_ecosystem"),
        ("CE-Ecosystem Builder (vLLM + Guidance)", "cahul_ce_ecosystem"),
    ]

    print(f"\n🔄 Updating {len(ecosystems)} ecosystems...")

    for framework, ecosystem in ecosystems:
        file_path = Path(f"/home/thiesen/Documents/AI-Innoscence_Ecosystem/{framework}/{ecosystem}/scrapers/batch_processor.py")

        if not file_path.exists():
            print(f"  ✗ {framework}/{ecosystem}: File not found")
            continue

        try:
            if replace_geocode_method(file_path, hamburg_method, ecosystem):
                print(f"  ✓ {framework}/{ecosystem}: Updated successfully")
            else:
                print(f"  ⚠️  {framework}/{ecosystem}: Could not update")
        except Exception as e:
            print(f"  ✗ {framework}/{ecosystem}: Error - {e}")

    print("\n✅ Geocoding fixes applied to all ecosystems!")


if __name__ == '__main__':
    main()
