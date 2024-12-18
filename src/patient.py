import random
from helper import get_response

class Patient:
    def __init__(self, args, sample):
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
        self.facts = sample['atomic_facts'] if 'atomic_facts' in sample else None  # To store atomic facts after initial processing, you can choose to store this somewhere locally to avoid repeated processing

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
        raise NotImplementedError
    

class RandomPatient(Patient):
    def respond(self, question):
        # Randomly select a response mode
        if random.random() < 0.5 or len(self.context_list) == 0:
            answer = "The patient cannot answer this question, please do not ask this question again."
        else:
            answer = random.choice(self.context_list)
        self.update_state(question, answer)
        return answer

class DirectPatient(Patient):
    def respond(self, question):
        system_prompt = "Answer the question with the given context."
        user_prompt = f"Context: \"{self.initial_info}\"\nQuestion: \"{question}\"\n"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = get_response(messages, max_length=50).strip()
        self.update_state(question, response)
        return response

class InstructPatient(Patient):
    def respond(self, question):
        system_prompt = "You are a truthful assistant that understands the patient's information, and you are trying to answer questions from a medical doctor about the patient."
        user_prompt = f"Below is a context paragraph describing the patient and their conditions:\n\"{self.context_para}\"\nQuestion from the doctor: \"{question}\"\nUse the context paragraph to answer the doctor's question. If the paragraph does not answer the question, simply say \"The patient cannot answer this question, please do not ask this question again.\" Answer only what the question asks for. Do not provide any analysis, inference, or implications. Respond with a straightforward answer to the question ONLY and NOTHING ELSE."
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = get_response(messages, max_length=50)[0]['generated_text']
        self.update_state(question, response)
        return response
    
class FactSelectPatient(Patient):
    def respond(self, question):
        if not self.facts:
            # Decompose context into facts if not already done
            system_prompt = "You are a truthful medical assistant that understands the patient's information."
            user_prompt = f"Break the following patient information into a list of independent atomic facts, with one piece of information in each statement. Each fact should only include the smallest unit of information, but should be self-contained.\n\"{self.context_para}\"\nResponse with the list of atomic facts and nothing else, prepend each fact by an index starting from 1. No sub-list allowed."
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            self.facts = get_response(messages, max_length=1000).splitlines()
        
        facts_prompt = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(self.facts))
        system_prompt = f"List of facts:\n{facts_prompt}\n\nQuestion from the doctor: \"{question}\"\nWhich of the above facts answer the question?"
        messages = [{"role": "system", "content": system_prompt}]
        response = get_response(messages, max_length=50)[0]['generated_text']
        self.update_state(question, response)
        return response