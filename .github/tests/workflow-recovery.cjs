const {readFileSync} = require('node:fs');
const {test} = require('node:test');
const assert = require('node:assert/strict');

// Execute the workflow's actual script, not a second copy of its decision rule.
const workflow = readFileSync('.github/workflows/retry-scheduling.yml', 'utf8');
const script = workflow.split('          script: |\n')[1].split('\n')
  .map(line => line.slice(12)).join('\n');
const runScript = new (Object.getPrototypeOf(async function() {}).constructor)(
  'github', 'context', 'core', script);
const base = {path: '.github/workflows/ci.yml', status: 'completed', conclusion: 'failure',
  run_attempt: 1, event: 'pull_request', head_branch: 'feature'};
async function check(run, jobs, changed) {
  const reruns = [], logs = [];
  let reads = 0;
  const github = {
    rest: {actions: {
      getWorkflowRun: async () => ({data: reads++ === 0 ? run : (changed || run)}),
      listJobsForWorkflowRun: Symbol('jobs'),
      reRunWorkflow: async params => reruns.push(params),
    }},
    paginate: async (method, params) => {
      assert.equal(method, github.rest.actions.listJobsForWorkflowRun);
      assert.equal(params.filter, 'latest');
      assert.equal(params.per_page, 100);
      return jobs;
    },
  };
  await runScript(github, {repo: {owner: 'OrbitInfra', repo: 'OrbitKV'},
    payload: {workflow_run: {id: 42}}}, {info: text => logs.push(text)});
  assert.ok(logs.length);
  return reruns.length;
}
const success = {name: 'passed', status: 'completed', conclusion: 'success'};
test('missing jobs and never-started jobs retry the whole first attempt once', async () => {
  for (const jobs of [[], Array(5).fill(success), [success,
    {status: 'queued', conclusion: null, started_at: null},
    {status: 'completed', conclusion: 'skipped'}]]) {
    assert.equal(await check(base, jobs), 1);
    assert.equal(await check({...base, run_attempt: 2}, jobs), 0);
  }
  assert.equal(await check({...base, event: 'push', head_branch: 'main'}, [success]), 1);
});
test('real failures, timeouts, cancellations and ambiguous jobs stay visible', async () => {
  for (const conclusion of ['failure', 'timed_out', 'cancelled', 'action_required', 'stale', 'neutral', 'unknown', null]) {
    assert.equal(await check(base, [success, {status: 'completed', conclusion}]), 0);
  }
  assert.equal(await check(base, [{status: 'in_progress', conclusion: null}]), 0);
  assert.equal(await check(base, [{status: 'queued', conclusion: null, started_at: '2026-11-10'}]), 0);
});
test('scope and concurrent reruns fail closed', async () => {
  for (const patch of [{event: 'workflow_dispatch'}, {event: 'push', head_branch: 'feature'},
    {path: '.github/workflows/release.yml'}, {conclusion: 'success'}, {status: 'in_progress'},
    {run_attempt: 2}, {run_attempt: undefined}]) {
    assert.equal(await check({...base, ...patch}, [success]), 0);
  }
  assert.equal(await check(base, [success], {...base, run_attempt: 2}), 0);
});
test('privileged workflow has only Actions write and executes no checkout or PR input', () => {
  assert.match(workflow, /permissions:\n  actions: write\n\n/);
  assert.doesNotMatch(workflow, /uses: actions\/(checkout|download-artifact)|contents:|pull_request_target/);
  assert.doesNotMatch(script, /\$\{\{/);
});
