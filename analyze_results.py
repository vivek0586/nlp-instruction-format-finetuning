"""Audit submitted Colab predictions and create report-ready summaries (stdlib only)."""
import json, zipfile, statistics, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent
z = zipfile.ZipFile(ROOT / 'colab_results.zip')
read = lambda n: json.loads(z.read(n))
runs = sorted({n.split('/')[1] for n in z.namelist() if n.startswith('results/')})
summary = {}
rows = []
reference = {}
for run in runs:
    for task, labels, expected in [('emotion', ['sadness','joy','love','anger','fear','surprise'],2000), ('sentiment',['negative','positive'],872)]:
        paired = {}
        for model in ['base','adapted']:
            stem = f'results/{run}/{task}_{model}'
            records = [json.loads(line) for line in z.read(stem+'_predictions.jsonl').decode().splitlines()]
            assert len(records) == expected
            identities = [(r['text'],r['gold']) for r in records]
            reference.setdefault(task,identities)
            assert identities == reference[task], (run,task,'example order mismatch')
            accuracy = sum(r['gold']==r['prediction'] for r in records)/expected
            invalid = sum(r['prediction'] not in labels for r in records)/expected
            f1=[]
            for label in labels:
                tp=sum(r['gold']==label and r['prediction']==label for r in records)
                fp=sum(r['gold']!=label and r['prediction']==label for r in records)
                fn=sum(r['gold']==label and r['prediction']!=label for r in records)
                f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
            metrics=read(stem+'_metrics.json')
            for k,v in [('accuracy',accuracy),('invalid_rate',invalid),('macro_f1',statistics.mean(f1))]:
                assert abs(metrics[k]-v)<1e-10,(run,task,model,k)
            paired[model]=records
            rows.append({'run':run,'task':task,'model':model,**{k:metrics[k] for k in ['accuracy','macro_f1','invalid_rate']}})
            if model=='adapted':
                bad=[r for r in records if r['prediction'] not in labels]
                summary[run+'/'+task]={'invalid_output_counts':collections.Counter(r['raw_generation'] for r in bad).most_common(15),'valid_only_accuracy':sum(r['gold']==r['prediction'] for r in records)/(expected-len(bad)) if expected>len(bad) else None}
        pair=read(f'results/{run}/{task}_paired_analysis.json')
        gains=sum(a['prediction']!=a['gold'] and b['prediction']==b['gold'] for a,b in zip(paired['base'],paired['adapted']))
        losses=sum(a['prediction']==a['gold'] and b['prediction']!=b['gold'] for a,b in zip(paired['base'],paired['adapted']))
        assert gains==pair['wrong_to_correct'] and losses==pair['correct_to_wrong']
groups={}
for fmt in ['natural','compact']:
    for task in ['emotion','sentiment']:
        for model in ['base','adapted']:
            records=[r for r in rows if r['run'].startswith(fmt) and r['task']==task and r['model']==model]
            groups[f'{fmt}/{task}/{model}']={k:{'mean':statistics.mean(r[k] for r in records),'sample_sd':statistics.stdev(r[k] for r in records)} for k in ['accuracy','macro_f1','invalid_rate']}
out={'audit':'All 24 prediction files match saved accuracy, macro-F1 and invalid rates; example identities/order and all 12 gain/loss counts match. This does not verify training provenance or dataset leakage.','runs':rows,'aggregates':groups,'invalid_analysis':summary,'environment':read('colab_run_info.json')}
(ROOT/'results_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(out['audit'])
for key,values in groups.items():
    print(key,' '.join(f'{k}={100*v["mean"]:.2f} +/- {100*v["sample_sd"]:.2f}' for k,v in values.items()))
for key,values in summary.items():
    if key.endswith('/sentiment'): print(key,values)
