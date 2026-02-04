import unittest
from src.synthetic.combinatorial import CombinatorialGenerator
from src.synthetic.generator import SyntheticGenerator

class TestGenerators(unittest.TestCase):
    def setUp(self):
        self.comb_gen = CombinatorialGenerator()
        self.llm_gen = SyntheticGenerator()

    def test_combinatorial_generator(self):
        name = "Công ty TNHH ABC"
        variants = self.comb_gen.generate(name)
        self.assertGreater(len(variants), 0)
        # Check if basic variants are present
        found_tnhh = any("tnhh" in v.lower() for v in variants)
        self.assertTrue(found_tnhh)

    def test_llm_generator_init(self):
        # Basic check to ensure it initializes
        self.assertIsNotNone(self.llm_gen.client)
        self.assertIn("glm", self.llm_gen.model)

if __name__ == "__main__":
    unittest.main()
