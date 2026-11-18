"""Compute paired format contrasts, class transitions, and valid-subset metrics."""
import json, zipfile, statistics
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
z=zipfile.ZipFile(P/'colab_results.zip')
labels=['sadness','joy','love','anger','fear','surprise']
seeds=[17,42,73]
def read(f,s,t,m):
    return [json.loads(x) for x in z.read(f'results/{f}_seed{s}/{t}_{m}_predictions.jsonl').decode().splitlines()]
def table(headers,rows,caption):
    return '\n'.join([r'\begin{center}\small',r'\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(headers)+r' \\ \midrule',*[' & '.join(map(str,r))+r' \\' for r in rows],r'\bottomrule\end{tabular}',r'\par\smallskip '+caption,r'\end{center}'])
contrast=[]; classrows=[]; matched=[]
for task in ['emotion','sentiment']:
    for seed in seeds:
        a=read('natural',seed,task,'adapted');b=read('compact',seed,task,'adapted')
        assert [(r['text'],r['gold']) for r in a]==[(r['text'],r['gold']) for r in b]
        d=np.array([int(x['prediction']==x['gold'])-int(y['prediction']==y['gold']) for x,y in zip(a,b)])
        # Resampling the empirical {-1,0,1} distribution is exactly equivalent
        # to an example-paired bootstrap for a difference in accuracies.
        prob=np.array([(d==v).mean() for v in [-1,0,1]])
        counts=np.random.default_rng(seed).multinomial(len(d),prob,size=10000)
        lo,hi=np.quantile((counts[:,2]-counts[:,0])/len(d),[.025,.975])*100
        contrast.append([task.title(),seed,f'{100*d.mean():.2f}',f'[{lo:.2f}, {hi:.2f}]'])
for label in labels:
    row=[label]
    for fmt in ['natural','compact']:
        gain=[];loss=[]
        for seed in seeds:
            a=read(fmt,seed,'emotion','base');b=read(fmt,seed,'emotion','adapted')
            gain.append(sum(x['gold']==label and x['prediction']!=x['gold'] and y['prediction']==y['gold'] for x,y in zip(a,b)))
            loss.append(sum(x['gold']==label and x['prediction']==x['gold'] and y['prediction']!=y['gold'] for x,y in zip(a,b)))
        row += [f'{statistics.mean(gain):.1f}',f'{statistics.mean(loss):.1f}']
    classrows.append(row)
for fmt in ['natural','compact']:
    for seed in seeds:
        a=read(fmt,seed,'sentiment','base');b=read(fmt,seed,'sentiment','adapted')
        subset=[(x,y) for x,y in zip(a,b) if y['prediction'] in ['negative','positive']]
        base=100*sum(x['prediction']==x['gold'] for x,y in subset)/len(subset)
        adapted=100*sum(y['prediction']==y['gold'] for x,y in subset)/len(subset)
        matched.append([fmt.title(),seed,len(subset),f'{base:.2f}',f'{adapted:.2f}'])
text=r'''\section{Additional Checks of Format Effects and Regressions}
The adaptation-choice question warrants a direct comparison of the two adapted models, beyond separate base-to-adapted gains. Table 6 pairs the same evaluation examples within each seed. Each Emotion interval includes zero, so the observed natural-format advantage is not distinguished from evaluation-sample variation by these individual intervals. SST-2 contrasts change sign across seeds; this again argues against a general format ranking. These are exploratory, unadjusted intervals, not a multiple-comparison significance claim.
'''
text+=table(['Task','Seed','Natural minus compact','95\\% paired CI'],contrast,'Table 6: Direct adapted-model accuracy contrasts in percentage points; 10,000 example resamples per seed. Training-format and evaluation-format effects remain coupled.')
text+=r'''
To avoid relying on the illustrative seed-42 transition counts, Table 7 summarizes gains and regressions across all seeds. These are average counts per run over the same 2,000 examples, not counts of distinct examples pooled across runs. Surprise and love still suffer regressions despite large average improvements, while fear has especially few regressions.
'''
text+=table(['Class','Natural gains','Natural losses','Compact gains','Compact losses'],classrows,'Table 7: Mean wrong-to-correct and correct-to-wrong counts across seeds 17, 42, and 73, relative to each matched base prompt.')
text+=r'''
Finally, comparing valid-only adapted accuracy to an all-example baseline changes the evaluation population. Table 8 instead evaluates both models on exactly the subset accepted from the adapted model. Natural adaptation remains worse on each matched subset; compact adaptation is better on these selected subsets. This diagnostic conditions on adapted behavior and cannot establish performance on excluded examples or recover the overall accuracy lost to invalid outputs.
'''
text+=table(['Format','Seed','Accepted n','Base accuracy','Adapted accuracy'],matched,'Table 8: SST-2 accuracy (percent) on identical subsets where the adapted output is a permitted sentiment label. Overall results in Table 1 retain all examples.')
(P/'original_seed_analysis.tex').write_text(text,encoding='utf-8')
(P/'rubric_analysis.json').write_text(json.dumps({'direct_format_contrasts':contrast,'class_transitions':classrows,'matched_valid_subsets':matched},indent=2))
print(text)
