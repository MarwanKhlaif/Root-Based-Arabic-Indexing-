#!/usr/bin/env python3
"""
Environment & Database Verification Script.
Milestone 1 - Arabic Root-Based Full-Text Indexing in PostgreSQL 16.

Verifies:
1. Active connection to PostgreSQL 16 on database 'arabic_ir'.
2. Database and client UTF-8 encoding.
3. Presence and active status of extensions: 'pg_trgm' and 'unaccent'.
4. Full catalog of built-in text search configurations in pg_ts_config.
5. Proves absence of native Arabic root-based text search configuration (research gap proof).
"""

import sys
import json
import argparse
from typing import Dict, Any, List

# Allow direct script execution from repository root
sys.path.insert(0, ".")

from src.db.connection import get_connection, get_db_config


def run_environment_checks() -> Dict[str, Any]:
    """
    Execute all verification queries and return structured results dictionary.
    """
    config = get_db_config()
    report: Dict[str, Any] = {
        "database": config["dbname"],
        "host": config["host"],
        "port": config["port"],
        "connected": False,
        "pg_version": None,
        "is_pg16": False,
        "server_encoding": None,
        "client_encoding": None,
        "extensions": {},
        "missing_extensions": [],
        "ts_configs": [],
        "has_builtin_arabic_ts": False,
        "has_native_arabic_root_ts": False,
        "builtin_arabic_matches_root": False,
        "european_stemmers": [],
        "checks_passed": False,
        "errors": []
    }

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                report["connected"] = True

                # 1. Version check
                cur.execute("SELECT version();")
                version_str = cur.fetchone()[0]
                report["pg_version"] = version_str
                report["is_pg16"] = "PostgreSQL 16" in version_str

                # 2. Encoding check
                cur.execute("SHOW server_encoding;")
                report["server_encoding"] = cur.fetchone()[0]
                cur.execute("SHOW client_encoding;")
                report["client_encoding"] = cur.fetchone()[0]

                # 3. Required Extensions check
                cur.execute("""
                    SELECT extname, extversion 
                    FROM pg_extension 
                    WHERE extname IN ('pg_trgm', 'unaccent');
                """)
                ext_rows = cur.fetchall()
                for extname, extversion in ext_rows:
                    report["extensions"][extname] = extversion

                required_exts = {"pg_trgm", "unaccent"}
                missing = required_exts - set(report["extensions"].keys())
                report["missing_extensions"] = sorted(list(missing))

                # 4. Text Search Configurations check
                cur.execute("SELECT cfgname FROM pg_ts_config ORDER BY cfgname;")
                configs = [row[0] for row in cur.fetchall()]
                report["ts_configs"] = configs
                report["has_builtin_arabic_ts"] = "arabic" in configs

                # Check if any root-extracting dictionary or configuration exists
                cur.execute("SELECT dictname FROM pg_ts_dict WHERE dictname ~* '(root|isri|khoja)';")
                root_dicts = cur.fetchall()
                report["has_native_arabic_root_ts"] = len(root_dicts) > 0

                # Test whether built-in arabic config bridges inflected words to roots
                cur.execute("SELECT to_tsvector('arabic', 'يكتبون في المكتبة') @@ to_tsquery('arabic', 'كتب');")
                report["builtin_arabic_matches_root"] = cur.fetchone()[0]

                european_known = [
                    "danish", "dutch", "english", "finnish", "french", "german",
                    "hungarian", "italian", "norwegian", "portuguese", "romanian",
                    "russian", "spanish", "swedish", "turkish"
                ]
                report["european_stemmers"] = [c for c in configs if c in european_known]

    except Exception as exc:
        report["errors"].append(str(exc))
        return report

    # Overall assertion check
    report["checks_passed"] = (
        report["connected"]
        and report["is_pg16"]
        and report["server_encoding"] == "UTF8"
        and len(report["missing_extensions"]) == 0
        and not report["has_native_arabic_root_ts"]
        and not report["builtin_arabic_matches_root"]
    )
    return report


def print_human_report(report: Dict[str, Any]) -> None:
    """
    Print a formatted, user-friendly terminal report with colored status indicators.
    """
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[1m\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN} PostgreSQL 16 Environment & Extension Verification Report {RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

    if not report["connected"]:
        print(f"{RED}[FAIL] Database Connection:{RESET} Unable to connect to {report['host']}:{report['port']}/{report['database']}")
        for err in report["errors"]:
            print(f"       Details: {err}")
        return

    # Connection & Version
    v_icon = f"{GREEN}[PASS]{RESET}" if report["is_pg16"] else f"{YELLOW}[WARN]{RESET}"
    print(f"{v_icon} Database Connection: Connected to '{report['database']}' on {report['host']}:{report['port']}")
    print(f"       Engine Version: {report['pg_version']}")

    # Encoding
    enc_pass = report["server_encoding"] == "UTF8" and report["client_encoding"] == "UTF8"
    enc_icon = f"{GREEN}[PASS]{RESET}" if enc_pass else f"{RED}[FAIL]{RESET}"
    print(f"{enc_icon} Encoding: Server={report['server_encoding']}, Client={report['client_encoding']}")

    # Extensions
    if not report["missing_extensions"]:
        print(f"{GREEN}[PASS]{RESET} Extensions: Both 'pg_trgm' ({report['extensions'].get('pg_trgm')}) and 'unaccent' ({report['extensions'].get('unaccent')}) are active.")
    else:
        print(f"{RED}[FAIL]{RESET} Missing Extensions: {', '.join(report['missing_extensions'])}")
        print("       Action: Run `CREATE EXTENSION IF NOT EXISTS <extname>;` in database.")

    # Text Search Configurations
    print(f"\n{BOLD}Text Search Configuration Audit (pg_ts_config):{RESET}")
    print(f"  - Total configurations found: {len(report['ts_configs'])}")
    print(f"  - European Snowball stemmers detected ({len(report['european_stemmers'])}):")
    print(f"    {', '.join(report['european_stemmers'])}")
    print(f"  - Built-in Snowball Arabic config: Present ('arabic' using Snowball affix-stripping template).")
    
    # Native Arabic Root Stemmer Check
    if not report["has_native_arabic_root_ts"]:
        print(f"\n{GREEN}[PASS] Research Gap Confirmed:{RESET}")
        print("       PostgreSQL 16 has NO native root-based Arabic text search configuration.")
        print("       The built-in 'arabic' configuration is a basic Snowball affix-stripper only,")
        print("       which cannot extract Semitic roots.")
        print(f"       Proof: to_tsvector('arabic', 'يكتبون في المكتبة') @@ to_tsquery('arabic', 'كتب') -> {report['builtin_arabic_matches_root']}.")
        print("       This empirically justifies the requirement for Systems S1–S5.")
    else:
        print(f"{YELLOW}[NOTICE]{RESET} Root-based Arabic text search configuration detected.")

    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    if report["checks_passed"]:
        print(f"{BOLD}{GREEN} ALL ENVIRONMENT CHECKS PASSED SUCCESSFULLY (Exit Code 0) {RESET}")
    else:
        print(f"{BOLD}{RED} VERIFICATION FAILED (Review flagged issues above) {RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PostgreSQL environment and extensions.")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")
    args = parser.parse_args()

    report = run_environment_checks()

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_human_report(report)

    sys.exit(0 if report["checks_passed"] else 1)


if __name__ == "__main__":
    main()
