"""
BED - Byte-Equivalent Decompilation
Implementation of genetic algorithm-based decompilation from Schulte et al. paper
"""

import subprocess
import tempfile
import random
import hashlib
import re
import os
import json
import difflib
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


@dataclass
class BEDConfig:
    """Configuration for BED evolutionary search"""
    pop_size: int = 1024
    max_evals: int = 131072
    cross_rate: float = 0.25
    target_chance: float = 0.75  # Probability of targeting mutations to bad regions
    num_frankensteins: int = 100
    fix_lit_chance: float = 0.1
    compiler: str = "gcc"
    compiler_flags: List[str] = field(default_factory=lambda: ["-m32", "-g", "-O0"])
    parallel_workers: int = 4
    tournament_size: int = 7
    use_lexicase: bool = True
    
    # mutation probabilities
    mutation_weights: Dict[str, float] = field(default_factory=lambda: {
        "cut": 0.1,
        "insert": 0.15,
        "swap": 0.1,
        "replace": 0.15,
        "fix_literals": 0.1,
        "promote_guarded": 0.05,
        "explode_for_loop": 0.05,
        "coalesce_while_loop": 0.05,
        "arith_assign_expansion": 0.05,
        "rename_variable": 0.1,
        "insert_from_db": 0.1,
    })

@dataclass
class Candidate:
    """Individual candidate decompilation"""
    source_code: str
    fitness: float = float('inf')
    binary_path: Optional[Path] = None
    disassembly: Optional[List[str]] = None
    diff_regions: List[Tuple[int, int]] = field(default_factory=list)
    instruction_matches: np.ndarray = field(default_factory=lambda: np.array([]))
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    id: str = field(default_factory=lambda: hashlib.md5(str(random.random()).encode()).hexdigest()[:8])

class CodeDatabase:
    """Database of code snippets for mining"""

    def __init__(self, db_path: Optional[str] = None):
        self.snippets = []
        self.function_snippets = []
        self.statement_snippets = []
        self.loop_snippets = []

        if db_path and Path(db_path).exists():
            self.load_database(db_path)
        else:
            self.generate_default_snippets()
        
    def generate_default_snippets(self):
        """Generate default code snippets for experiments"""

        # basic funcs
        self.function_snippets = [
            """int compute(int n) {
                int result = 0;
                for (int i = 0; i < n; i++) {
                    result += i;
                }
                return result;
            }""",
            """void process(int* arr, int size) {
                for (int i = 0; i < size; i++) {
                    arr[i] = arr[i] * 2;
                }
            }""",
            """int fibonacci(int n) {
                if(n <= 1) return n;
                return fibonacci(n-1) + fibonacci(n-2);
            }""",
        ]

        # common statements
        self.statement_snippets = [
            "sum += i;",
            "count++;",
            "if(x % 2 == 0) { total += x; }",
            "printf(\"%d\\n\", result);",
            "temp = a; a = b; b = temp;",
        ]

        # loop patterns
        self.loop_snippets = [
            "for(int i = 0; i < n; i++) { }",
            "while(n > 0) { n--; }",
            "do { } while(condition);",
        ]
    
    def search_similar(self, target_bytes: bytes, limit: int = 10) -> List[str]:
        """Search for snippets similar to target bytes"""
        # simplified similarity search
        results = random.sample(self.statement_snippets + self.loop_snippets, min(limit, len(self.statement_snippets + self.loop_snippets)))
        return results
    
    def get_random_function(self) -> str:
        """Get random function from database"""
        if self.function_snippets:
            return random.choice(self.function_snippets)
        return "int dummpy() { return 0; }"
    
    def get_random_statement(self) -> str:
        """Get random statement from database"""
        if self.statement_snippets:
            return random.choice(self.statement_snippets)
        return "x = x + 1;"

class Compiler:
    """Handles compilation and binary analysis"""

    def __init__(self, config: BEDConfig):
        self.config = config
        self.compile_cache = {}

    def compile(self, source_code: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Compile source code and return (success, binary_path, error_msg)"""
        # Check cache
        code_hash = hashlib.md5(source_code.encode()).hexdigest()
        if code_hash in self.compile_cache:
            return self.compile_cache[code_hash]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as src_file:
            src_file.write(source_code)
            src_path = src_file.name
        
        out_path = src_path.replace('.c', '.out')

        try:

            cmd = [self.config.compiler] + self.config.compiler_flags + ['-o', out_path, src_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                self.compile_cache[code_hash] = (True, out_path, None)
                return True, out_path, None
            else:
                error_msg = result.stderr
                return False, None, error_msg
            
        except subprocess.TimeoutExpired:
            return False, None, "Compilation timed out"
        except Exception as e:
            return False, None, str(e)
        finally:
            if os.path.exists(src_path):
                os.unlink(src_path)
    
    def disassemble(self, binary_path: str) -> List[str]:
        """Disassemble binary and return list of instructions"""
        try:
            result = subprocess.run(['objdump', '-d', binary_path], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                # Parse disassembly output
                instructions = []
                for line in result.stdout.split('\n'):
                    # extract actual instruction lines
                    if re.match(r'^\s*[0-9a-f]+:', line):
                        instructions.append(line.strip())
                return instructions
            
        except:
            pass
        return []
    
    def extract_literals(self, binary_path: str) -> Dict[str, Any]:
        """Extract literals from binary"""
        literals = {
            'strings': [],
            'integers': [],
            'floats': []
        }

        try:
            # Extract strinsg
            result = subprocess.run(['strings', binary_path], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                literals['strings'] = [s.strip() for s in result.stdout.split('\n') if s.strip()]
            
            # Extract constants from disassembly
            disasm = self.disassemble(binary_path)
            for line in disasm:

                # looking for immediate values
                if '$0x' in line:
                    match = re.search(r'\$0x([0-9a-f]+)', line)
                    if match:
                        value = int(match.group(1), 16)
                        if value not in literals['integers']:
                            literals['integers'].append(value)
                elif '$' in line:
                    match = re.search(r'\$(\d+)', line)
                    if match:
                        value = int(match.group(1))
                        if value not in literals['integers']:
                            literals['integers'].append(value)
        
        except:
            pass

        return literals
    
class FitnessEvaluator:
    """Evaluates fitness of candidates decompilation"""

    def __init__(self, target_binary: str, compiler: Compiler):
        self.target_binary
        self.compiler = compiler
        self.target_disasm = compiler.disassemble(target_binary)
        self.target_literals = compiler.extract_literals(target_binary)
    
    def evaluate(self, candidate: Candidate) -> Tuple[float, List[Tuple[int, int]], np.array]:
        """
        Evaluate candidate fitness against target binary
        Returns: (fitness_score, diff_regions, instruction_matches)
        """
        # compile candidate
        success, binary_path, error_msg = self.compiler.compile(candidate.source_code)

        if not success:
            # compilation failed, assign worst fitness
            return float('inf'), [(0, len(self.target_disasm))], np.zeros(len(self.target_disasm))
        
        candidate.binary_path = binary_path

        # get disasm
        candidate_disasm = self.compiler.disassemble(binary_path)
        candidate.disassembly = candidate_disasm

        # compute similarity
        fitness, diff_regions, matches = self._compute_similarity(candidate_disasm)

        return fitness, diff_regions, matches
    
    def _compute_similarity(self, candidate_disasm: List[str]) -> Tuple[float, List[Tuple[int, int]], np.array]:
        """Compute byte similarity between target and candidate disassembly"""
        # will use the difflib sequence matcher for similarity
        matcher = difflib.SequenceMatcher(None, self.target_disasm, candidate_disasm)

        # track matches per instruct 
        matches = np.zeros(len(self.target_disasm))
        diff_regions = []

        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'equal':
                matches[i1:i2] = 1
            else:
                diff_regions.append((i1, i2))
        
        # fitness is proportion of unmatched instructions
        fitness = 1.0 - (np.sum(matches) / len(self.target_disasm))

        return fitness, diff_regions, matches

class MutationOperator:
    """Handles source-to-source mutations"""

    def __init__(self, code_db: CodeDatabase, fitness_eval: FitnessEvaluator):
        self.code_db = code_db
        self.fitness_eval = fitness_eval
    
    def mutate(self, candidate: Candidate, mutation_type: Optional[str] = None) -> Candidate:
        """Apply mutation to candidate"""

        if mutation_type is None:
            mutation_type = self._select_mutation_type()

        new_source = self._apply_mutation(candidate.source_code, mutation_type, candidate)

        new_candidate = Candidate(
            source_code=new_source,
            generation=candidate.generation + 1,
            parent_ids=[candidate.id]
        )

        return new_candidate
    
    def _select_mutation_type(self, candidate: Candidate) -> str:
        """Select mutation type based on candidate state"""
        # If candidate has diff regions, prefer targeted mutations
        if candidate.diff_regions and random.random() < 0.75:
            return random.choice(['fix_literals', 'replace', 'insert_from_db'])
        else:
            return random.choice(list(BEDConfig().mutation_weights.keys()))
    