"""
Validator CLI for avatar talent kits.
Validates talent.json schemas and checks required files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from .talent_kit import TalentKit


def validate_single_kit(talent_kit: TalentKit, talent_id: str, verbose: bool = False) -> bool:
    """Validate a single talent kit."""
    print(f"Validating kit: {talent_id}")
    
    validation = talent_kit.validate_kit(talent_id)
    
    if validation["valid"]:
        print(f"✅ {talent_id} is valid")
        if verbose and validation["warnings"]:
            for warning in validation["warnings"]:
                print(f"  ⚠️  {warning}")
        return True
    else:
        print(f"❌ {talent_id} is invalid")
        for error in validation["errors"]:
            print(f"  ❌ {error}")
        if validation["warnings"]:
            for warning in validation["warnings"]:
                print(f"  ⚠️  {warning}")
        return False


def validate_all_kits(talent_kit: TalentKit, verbose: bool = False) -> Dict[str, Any]:
    """Validate all talent kits."""
    print("Validating all talent kits...")
    
    kits = talent_kit.list_kits()
    
    if not kits:
        print("No talent kits found")
        return {"valid_count": 0, "total_count": 0, "kits": []}
    
    valid_count = 0
    results = []
    
    for kit in kits:
        is_valid = kit["valid"]
        if is_valid:
            valid_count += 1
        
        results.append({
            "id": kit["id"],
            "valid": is_valid,
            "errors": kit["errors"],
            "warnings": kit["warnings"]
        })
        
        if verbose or not is_valid:
            status = "✅" if is_valid else "❌"
            print(f"{status} {kit['id']}")
            if kit["errors"]:
                for error in kit["errors"]:
                    print(f"    ❌ {error}")
            if kit["warnings"]:
                for warning in kit["warnings"]:
                    print(f"    ⚠️  {warning}")
    
    print(f"\nValidation complete: {valid_count}/{len(kits)} kits are valid")
    
    return {
        "valid_count": valid_count,
        "total_count": len(kits),
        "kits": results
    }


def main():
    parser = argparse.ArgumentParser(description="Validate avatar talent kits")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), 
                       help="Base directory for talent factory")
    parser.add_argument("--talent-id", type=str, 
                       help="Specific talent ID to validate")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show warnings and detailed output")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")
    
    args = parser.parse_args()
    
    talent_kit = TalentKit(args.base_dir)
    
    if args.talent_id:
        # Validate single kit
        is_valid = validate_single_kit(talent_kit, args.talent_id, args.verbose)
        if args.json:
            result = {
                "talent_id": args.talent_id,
                "valid": is_valid
            }
            print(json.dumps(result, indent=2))
        sys.exit(0 if is_valid else 1)
    else:
        # Validate all kits
        results = validate_all_kits(talent_kit, args.verbose)
        if args.json:
            print(json.dumps(results, indent=2))
        sys.exit(0 if results["valid_count"] == results["total_count"] else 1)


if __name__ == "__main__":
    main()
