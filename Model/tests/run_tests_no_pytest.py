"""
Zero-dependency test runner used ONLY because pytest isn't installable in
this offline sandbox. Once you `pip install -r requirements.txt` in a
normal environment, just run `pytest tests/` instead -- these test files
are plain pytest-style test functions and work fine under real pytest.
"""
import sys, os, traceback, importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TEST_MODULES = [
    "tests.test_arabic_normalize",
    "tests.test_nid_validator",
    "tests.test_image_normalization",
    "tests.test_localization_scale_invariance",
]

passed, failed = 0, []
for mod_name in TEST_MODULES:
    mod = importlib.import_module(mod_name)
    for name in dir(mod):
        if name.startswith("test_"):
            fn = getattr(mod, name)
            try:
                fn()
                passed += 1
                print(f"PASS  {mod_name}.{name}")
            except Exception:
                failed.append((mod_name, name))
                print(f"FAIL  {mod_name}.{name}")
                traceback.print_exc()

print(f"\n{passed} passed, {len(failed)} failed")
if failed:
    sys.exit(1)
