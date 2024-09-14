import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250)
dspy.settings.configure(lm=turbo)

gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]


if __name__ == "__main__":
    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.prog(question=question)

    config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

    teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
    optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)

    evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

    evaluate(optimized_cot)

    turbo.inspect_history(n=1)
    optimized_cot("a woman is driving her son to the hospital and dies in a car crash. her son is injured and rushed to the hospital. the doctor yells, 'i can't operate on this boy, because he is my son.' how is this possible?")
