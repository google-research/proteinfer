In one terminal, run from the folder `interactive_manuscript`:
```bash ./compile.sh```
(You may need to install fswatch with `sudo apt-get install fswatch`.)

In another, run
```python -m SimpleHTTPServer 8888```
from the repo root.

In your browser, you can go to
`localhost:8888/interactive_manuscript/index.html`
if you're running on your desktop or you can go to, e.g.,
`your_desktop_hostname:8888/interactive_manuscript/index.html`
if you're working remotely.

To push to origin (due to the fact that node modules have public keys in them, which is outside our control):

```git push -o nokeycheck sso://google-brain/_direct/mlbileschi-proteinfer master```

Please do not force push.
