import json
import pandas as pd
from loguru import logger
import glob

# Load evaluation results
eval_dirs = ["WebVoyager30/LumiTeh/2854009812"]
# eval_dirs = ["WebVoyager30/BrowserUse/3956007623"]
# eval_dirs = ["WebVoyager30/Convergence/4847118934"]

def fetch_results(eval_dirs):
    results = []
    for directory in eval_dirs:
        for file in sorted(glob.glob(f"{directory}/**/results_no_screenshot.json", recursive=True)):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    dataset = directory.split('/')[0]
                    provider = directory.split('/')[1]
                    timestamp = directory.split('/')[-1]

                    record = {
                        'uid': timestamp + "-" + str(data['run_id']),
                        'fatal_crash': 1 if data['eval'] is None else 0,
                        'agent_score': 'success' if data['success'] else 'failure',
                        'agent_answer': data['agent_answer'],
                        'steps': data['steps'],
                        'dataset': dataset,
                        'provider': provider,
                        'run_id': data['run_id'],
                        'task_id': data['task']['id'],
                        'task': data['task']['question'],
                        'timestamp': timestamp,
                        'summary_file': file,
                        'webp_file': file.replace('results_no_screenshot.json', 'summary.webp'),
                        'duration': round(data['duration_in_s'], 0),
                    }

                    if data['eval'] is not None:
                        record.update({
                            'eval_score': data['eval']['eval'],
                            'eval_reason': data['eval']['reason']
                        })

                    results.append(record)

            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

    logger.info(f"Fetched {len(results)} evaluation results in total")
    return results

all_evals = fetch_results(eval_dirs)
finished_evals = [x for x in all_evals if x['fatal_crash'] == 0]
crashed_evals = [x for x in all_evals if x['fatal_crash'] == 1]
evals = finished_evals.copy()
logger.info(f"Valid evals: {len(finished_evals)} | Crashed evals: {len(crashed_evals)}")

unique_tasks = set(x['task_id'] for x in all_evals)
unique_uids = set(x['uid'] for x in all_evals)
logger.info(f"Unique tasks: {len(unique_tasks)} | Unique uids: {len(unique_uids)}")

# Average over runs then tasks
success_per_task = {}
for ev in finished_evals:
    if ev['task_id'] not in success_per_task:
        success_per_task[ev['task_id']] = {'agent_success': 0, 'eval_success': 0, 'total_runs': 0}
    success_per_task[ev['task_id']]['total_runs'] += 1
    success_per_task[ev['task_id']]['agent_success'] += 1 if ev['agent_score'] == 'success' else 0
    success_per_task[ev['task_id']]['eval_success'] += 1 if ev.get('eval_score') == 'success' else 0

for task_id, v in success_per_task.items():
    v['asr'] = '{:.3f}'.format(v['agent_success'] / v['total_runs'])
    v['esr'] = '{:.3f}'.format(v['eval_success'] / v['total_runs'])
    v.pop('agent_success')
    v.pop('eval_success')
    v.pop('total_runs')

r_tasks = dict(sorted(success_per_task.items(), key=lambda x: float(x[1]['esr']), reverse=True))
df_tasks = pd.DataFrame.from_dict(r_tasks, orient='index')
display(df_tasks)

asr_list = [float(v['asr']) for v in r_tasks.values()]
esr_list = [float(v['esr']) for v in r_tasks.values()]
avg_asr = '{:.3f}'.format(sum(asr_list) / len(asr_list))
avg_esr = '{:.3f}'.format(sum(esr_list) / len(esr_list))
print(f"Agent avg: {avg_asr} | Eval avg: {avg_esr} | {len(unique_uids)} runs")
