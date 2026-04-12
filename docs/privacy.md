# Privacy And Scalability Notes

## Privacy

A-Eye treats images as transient analysis inputs rather than long lived user
content.

- The extension sends only the selected image URL to the backend.
- The backend fetches the image, analyzes it in memory, and returns structured
  scores plus signal details.
- No database persistence is required for the current prototype.
- Input validation, payload size limits, and short request timeouts reduce
  unnecessary exposure and help avoid abuse.

## Scalability
The project is being shaped for containerized deployment so the backend can
scale independently from the browser client.

- FastAPI provides a clean API layer for horizontal scaling behind a load
  balancer later.
- Docker keeps backend dependencies reproducible across development and
  deployment environments.
- The current pipeline is intentionally modular so the CNN, feature analysis,
  and aggregation layers can evolve independently.

## Near-Term Follow-Up
- Replace the CNN inference stub with exported baseline model weights.
- Add request logging that excludes raw image content.
- Introduce rate limiting once the service moves beyond local development.
