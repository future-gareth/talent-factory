"""
Indexer CLI for avatar talent kits.
Scans /talents/avatar/ and returns JSON array of valid kits.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from .talent_kit import TalentKit


def index_talents(talent_kit: TalentKit, include_invalid: bool = False) -> List[Dict[str, Any]]:
    """Index all talent kits and return their information."""
    kits = talent_kit.list_kits()
    
    indexed = []
    for kit in kits:
        if kit["valid"] or include_invalid:
            kit_info = {
                "id": kit["id"],
                "path": kit["path"],
                "valid": kit["valid"]
            }
            
            if kit["manifest"]:
                kit_info.update({
                    "category": kit["manifest"].get("category"),
                    "kind": kit["manifest"].get("kind"),
                    "sdx_version": kit["manifest"].get("sdx_version"),
                    "base_model": kit["manifest"].get("base_model"),
                    "lora_rank": kit["manifest"].get("lora_rank"),
                    "default_weight": kit["manifest"].get("default_weight"),
                    "size_mb": kit["manifest"].get("size_mb"),
                    "created_at": kit["manifest"].get("created_at"),
                    "token": kit["manifest"].get("token"),
                    "negatives": kit["manifest"].get("negatives"),
                    "intended_mix": kit["manifest"].get("intended_mix", []),
                    "metadata": kit["manifest"].get("metadata", {})
                })
            
            if not kit["valid"]:
                kit_info["errors"] = kit["errors"]
                kit_info["warnings"] = kit["warnings"]
            
            indexed.append(kit_info)
    
    return indexed


def main():
    parser = argparse.ArgumentParser(description="Index avatar talent kits")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), 
                       help="Base directory for talent factory")
    parser.add_argument("--include-invalid", action="store_true",
                       help="Include invalid kits in the index")
    parser.add_argument("--output", "-o", type=Path,
                       help="Output file (default: stdout)")
    parser.add_argument("--pretty", action="store_true",
                       help="Pretty print JSON output")
    
    args = parser.parse_args()
    
    talent_kit = TalentKit(args.base_dir)
    
    try:
        indexed = index_talents(talent_kit, args.include_invalid)
        
        if args.pretty:
            output = json.dumps(indexed, indent=2)
        else:
            output = json.dumps(indexed)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Indexed {len(indexed)} talent kits to {args.output}")
        else:
            print(output)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"Error indexing talents: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
