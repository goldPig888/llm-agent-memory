# User service templates

Replace `@PROJECT_ROOT@`, `@PYTHON@`, and `@CAMPAIGN_ROOT@` in the `.in` files
with absolute paths. `@PYTHON@` is the Python executable in the installed
experiment environment. If the model is stored outside the default location,
also add `Environment=MUSE_ROOT=/absolute/model/storage` to each service.

Install the rendered files under `~/.config/systemd/user/` without the `.in`
suffix, then run:

```bash
systemctl --user daemon-reload
systemctl --user enable --now glimmer-baseline-watch.timer
systemctl --user enable --now glimmer-baseline-viewer.service
```

The watchdog checks every minute. A missing controller can be restarted;
orphaned but progressing workers are allowed to finish. A request with no new
request/response record for 45 minutes is treated as stalled. Interrupted
attempts are archived, completed results are preserved, and retries are bounded.
Three restart attempts within an hour require inspection. The viewer warns if
the monitoring record is more than 150 seconds old.

User services need an active user manager. On the experiment machine,
`loginctl enable-linger <username>` was enabled so monitoring survives logout.
Stop the timer explicitly before intentionally stopping or changing the campaign:

```bash
systemctl --user disable --now glimmer-baseline-watch.timer
```

The watchdog may start transient `glimmer-baseline-model-recovery` and
`glimmer-baseline-queue-recovery` services. Inspect logs with
`journalctl --user -u glimmer-baseline-watch.service`.
