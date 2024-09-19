import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=250)
dspy.settings.configure(lm=turbo)

gms8k = GSM8K()

trainset, devset = gms8k.train, gms8k.dev


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


evaluate = Evaluate(devset=devset[:],
                    metric=gsm8k_metric,
                    num_threads=4,
                    display_progress=True,
                    display_table=False)

cot_baseline = CoT()

evaluate(cot_baseline, devset=devset[:])

optimizer = BootstrapFewShotWithRandomSearch(
    metric=gsm8k_metric,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
)

cot_compiled = optimizer.compile(CoT(), trainset=trainset, valset=devset)

cot_compiled.save('iu_demo.json')
