import random
from helper import get_response

class Patient:
    def __init__(self, args, sample):
        self.variant = args.patient_variant
        # Assuming 'context' is a list or a long string of historical or background information
        if isinstance(sample['context'], list) and len(sample['context']) > 0:
            self.initial_info = sample['context'][0]  # Taking the first item if it's a list
            self.context_list = sample['context']
            self.context_para = ". ".join(sample['context'])
        elif isinstance(sample['context'], str):
            # Assuming sentences are separated by periods, taking the first sentence
            self.initial_info = sample['context'].split(". ")[0]
            self.context_list = sample['context'].split(". ")
            self.context_para = sample['context']
        else:
            self.initial_info = ""  # Default fallback
            self.context_list = []
        
        self.history = []  # To track the interaction history of questions and answers
        self.facts = None  # To store atomic facts after initial processing

    def update_state(self, question, answer):
        # Update the internal history with the new question and the corresponding answer
        self.history.append({"question": question, "answer": answer})

    def get_state(self):
        # Return the initial context and the history of interactions
        return {
            "initial_info": self.initial_info,
            "interaction_history": self.history
        }
    
    def get_questions(self):
        # Return the list of questions asked so far
        return [qa["question"] for qa in self.history]
    
    def get_answers(self):
        # Return the list of answers provided so far
        return [qa["answer"] for qa in self.history]
    
    def respond(self, question):
        if self.variant == 'direct':
            return self.respond_direct(question)
        elif self.variant == 'instruct':
            return self.respond_instruct(question)
        elif self.variant == 'fact_select':
            return self.respond_fact_select(question)
        elif self.variant == 'random':
            return self.respond_random(question)
        else:
            raise ValueError("Invalid response mode specified.")
        
    
    def respond_direct(self, question):
        system_prompt = "Answer the question with the given context."
        user_prompt = f"Context: \"{self.initial_info}\"\nQuestion: \"{question}\"\n"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = get_response(messages, max_length=50).strip()
        return response

    def respond_instruct(self, question):
        system_prompt = "You are a truthful assistant that understands the patient's information, and you are trying to answer questions from a medical doctor about the patient."
        user_prompt = f"Below is a context paragraph describing the patient and their conditions:\n\"{self.context_para}\"\nQuestion from the doctor: \"{question}\"\nUse the context paragraph to answer the doctor's question. If the paragraph does not answer the question, simply say \"The patient cannot answer this question, please do not ask this question again.\" Answer only what the question asks for. Do not provide any analysis, inference, or implications. Respond with a straightforward answer to the question ONLY and NOTHING ELSE."
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.model(messages, max_length=50)[0]['generated_text']
        return response.strip()


    def respond_fact_select(self, question):
        if not self.facts:
            # Decompose context into facts if not already done
            system_prompt = "You are a truthful medical assistant that understands the patient's information."
            user_prompt = f"Break the following patient information into a list of independent atomic facts, with one piece of information in each statement. Each fact should only include the smallest unit of information, but should be self-contained.\n\"{self.context_para}\"\nResponse with the list of atomic facts and nothing else, prepend each fact by an index starting from 1. No sub-list allowed."
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            self.facts = self.model(messages, max_length=1000).splitlines()
        
        facts_prompt = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(self.facts))
        system_prompt = f"List of facts:\n{facts_prompt}\n\nQuestion from the doctor: \"{question}\"\nWhich of the above facts answer the question?"
        messages = [{"role": "system", "content": system_prompt}]
        response = self.model(messages, max_length=50)[0]['generated_text']
        return response.strip()
    
    def respond_random(self, question):
        # Randomly select a response mode
        if random.random() < 0.5 or len(self.context_list) == 0:
            answer = "The patient cannot answer this question, please do not ask this question again."
        else:
            answer = random.choice(self.context_list)
        self.update_state(question, answer)
        return answer