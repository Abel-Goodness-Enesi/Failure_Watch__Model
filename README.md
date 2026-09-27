# Failure Watch

Flask version of the Failure Watch condition monitor. The neural network forward pass
runs in plain Python on the server (see `predict()` in `app.py`), using the same
trained weights as the original page. The front end just renders the page and polls
`/api/reading/<idx>` for each new sensor reading.

## Run locally

```
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## Hosting it publicly

Yes, this can be hosted publicly. Since it now has a Python backend, it needs a host
that runs a Python process, not a static file host like GitHub Pages. Reasonable free
or low cost options:

- **Render** (render.com): connect a GitHub repo, pick "Web Service", it detects Flask
  automatically. Free tier sleeps after inactivity.
- **Railway** (railway.app): similar flow, connect the repo, it builds and deploys.
- **PythonAnywhere**: good for small always-on Flask apps, simple free tier.
- **Fly.io**: a bit more setup but fast and reliable, has a free allowance.

Any of these will give you a public URL. Before deploying, make sure `app.py` binds to
`0.0.0.0` and reads the port from the `PORT` environment variable, which it already does.

## Files

- `app.py` - Flask app, model forward pass, and the two routes
- `model_data.json` - the trained weights, scaler values, and the 30 held out feed rows
- `templates/index.html` - the page, same layout and styling as the original
- `requirements.txt` - just Flask
