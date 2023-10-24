
Home -> Dashboards -> Gitlab -> GitLab CI pipelines 

In Query Tab Panel:
B Metrics browser:
```
-max(time() - gitlab_ci_pipeline_timestamp{project=~"$PROJECT", ref=~"$REF"}) by (project, ref, kind) unless max(gitlab_ci_pipeline_status{status=~"success", project=~"$PROJECT", ref=~"$REF"}) by (project, ref, kind) > 0
```
Legend: Verbose
Format: Table
Type: Instant


C Metrics browser:
```
max(gitlab_ci_pipeline_duration_seconds{project=~"$PROJECT",ref=~"$REF"}) by (project, ref, kind) unless (max(gitlab_ci_pipeline_status{status=~"success", project=~"$PROJECT",ref=~"$REF"}) by (project, ref, kind) > 0)
```
Legend: Verbose
Format: Table
Type: Instant


A Metrics browser:
```
(max(gitlab_ci_pipeline_status{project=~"$PROJECT",ref=~"$REF", status=~"running"}) by (project, ref, kind) * 2) > 0 or
(max(gitlab_ci_pipeline_status{project=~"$PROJECT",ref=~"$REF", status=~"failed"}) by (project, ref, kind) * 3) > 0 or
(max(gitlab_ci_pipeline_status{project=~"$PROJECT",ref=~"$REF", status=~"canceled"}) by (project, ref, kind) * 4) > 0 or
(max(gitlab_ci_pipeline_status{project=~"$PROJECT",ref=~"$REF", status=~"created"}) by (project, ref, kind) * 5) > 0 or
(max(gitlab_ci_pipeline_status{project=~"$PROJECT",ref=~"$REF", status=~"waiting_for_resource"}) by (project, ref, kind) * 6) > 0 or
(max(gitlab_ci_pipeline_status{project=~"$PROJECT",ref=~"$REF", status=~"preparing"}) by (project, ref, kind) * 7) > 0 or
(max(gitlab_ci_pipeline_status{project=~"$PROJECT",ref=~"$REF", status=~"pending"}) by (project, ref, kind) * 8) > 0 or
(max(gitlab_ci_pipeline_status{project=~"$PROJECT",ref=~"$REF", status=~"skipped"}) by (project, ref, kind) * 9) > 0 or
(max(gitlab_ci_pipeline_status{project=~"$PROJECT",ref=~"$REF", status=~"manual"}) by (project, ref, kind) * 10) > 0 or
(max(gitlab_ci_pipeline_status{project=~"$PROJECT",ref=~"$REF", status=~"scheduled"}) by (project, ref, kind) * 11) > 0
```


D Metrics browser:
```
max(gitlab_ci_pipeline_id{project=~"$PROJECT", ref=~"$REF"}) by (project, ref, kind, job_name) unless (max(gitlab_ci_pipeline_status{status=~"success", project=~"$PROJECT", ref=~"$REF"}) by (project, ref, kind, job_name) > 0)
```

In Transform Tab Panel:
Organize fields:
Value #D: ID
project: Project
kind: Ref Kind
ref: Ref Name
Value #B: Date
Value #C: Duration
Value #A: Status