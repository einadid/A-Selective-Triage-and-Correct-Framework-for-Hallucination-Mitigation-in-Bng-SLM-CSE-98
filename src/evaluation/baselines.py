"""
Baselines for SETU comparison
(a) raw SLM, no mitigation
(b) chain-of-thought prompting
(c) uniform RAG (retrieve-then-answer, no triage)
(d) uniform self-verification (chain-of-verification)
"""

class BaselineRunner:
    def __init__(self, slm_generator, retriever=None, nli_model=None):
        self.slm = slm_generator
        self.retriever = retriever
        self.nli = nli_model

    def raw_slm(self, query: str, context: str = None) -> str:
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer in Bengali:" if context else query
        return self.slm.generate(prompt, max_new_tokens=512, temperature=0.7)

    def chain_of_thought(self, query: str, context: str = None) -> str:
        cot_prompt = f"""প্রশ্ন: {query}
{'প্রসঙ্গ: ' + context if context else ''}

ধাপে ধাপে চিন্তা করো (Chain-of-Thought):
1. প্রশ্নটি বুঝো
2. প্রাসঙ্গিক তথ্য মনে করো
3. যৌক্তিকভাবে উত্তর দাও

চিন্তা প্রক্রিয়া এবং চূড়ান্ত উত্তর বাংলায় দাও:

উত্তর:"""
        return self.slm.generate(cot_prompt, max_new_tokens=512, temperature=0.7)

    def uniform_rag(self, query: str, context: str = None) -> str:
        # Retrieve then answer - no triage
        if self.retriever is None:
            return self.raw_slm(query, context)

        retrieval = self.retriever.retrieve_with_fallback(query, top_k=5)
        evidence = "\n".join([f"- {r['passage'][:400]}" for r in retrieval["results"][:3]])

        rag_prompt = f"""নিচের প্রমাণ ব্যবহার করে প্রশ্নের উত্তর দাও:

প্রমাণ:
{evidence}

প্রশ্ন: {query}
{'প্রসঙ্গ: ' + context if context else ''}

প্রমাণের ভিত্তিতে বাংলায় উত্তর দাও:
উত্তর:"""
        return self.slm.generate(rag_prompt, max_new_tokens=512, temperature=0.5)

    def chain_of_verification(self, query: str, context: str = None) -> str:
        # CoVe: answer, then verify, then revise
        # Step 1: initial answer
        initial = self.raw_slm(query, context)

        # Step 2: generate verification questions
        verify_prompt = f"""প্রশ্ন: {query}
প্রাথমিক উত্তর: {initial}

এই উত্তর যাচাই করার জন্য ৩টি যাচাই প্রশ্ন তৈরি করো:

যাচাই প্রশ্ন:"""
        verification_qs = self.slm.generate(verify_prompt, max_new_tokens=256, temperature=0.5)

        # Step 3: answer verification questions
        verify_answers_prompt = f"""যাচাই প্রশ্ন:
{verification_qs}

এই প্রশ্নগুলোর উত্তর দাও:

উত্তর:"""
        verification_answers = self.slm.generate(verify_answers_prompt, max_new_tokens=256, temperature=0.5)

        # Step 4: revise
        revise_prompt = f"""প্রশ্ন: {query}
প্রাথমিক উত্তর: {initial}
যাচাই প্রশ্ন: {verification_qs}
যাচাই উত্তর: {verification_answers}

যাচাইয়ের ভিত্তিতে প্রাথমিক উত্তর সংশোধন করো। চূড়ান্ত উত্তর বাংলায় দাও:

চূড়ান্ত উত্তর:"""
        final = self.slm.generate(revise_prompt, max_new_tokens=512, temperature=0.3)
        return final

    def run_all_baselines(self, query: str, context: str = None) -> dict:
        return {
            "raw_slm": self.raw_slm(query, context),
            "cot": self.chain_of_thought(query, context),
            "uniform_rag": self.uniform_rag(query, context),
            "cove": self.chain_of_verification(query, context)
        }
