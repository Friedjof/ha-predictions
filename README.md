# HA Predictions

[![License][license-shield]](LICENSE)
[![hacs][hacsbadge]][hacs]

A Home Assistant custom integration that uses machine learning to predict entity states based on feature entities. Train models to predict when lights turn on/off, switches change state, or other automations trigger based on the state of other entities in your home.

> ⚠️ This integration is in active development. Features may change, and there may be bugs or incomplete functionality. Use at your own risk and please report any issues you encounter.

## Features

- 🤖 **Machine Learning Integration**: Uses logistic regression to learn patterns from your Home Assistant entities
- 📊 **Training Mode**: Collect data from your entities to build prediction models
- 🎯 **Production Mode**: Make real-time predictions based on trained models
- 📈 **Performance Monitoring**: Track model accuracy and dataset size
- 🔄 **Flexible Configuration**: Choose target entity and any number of feature entities
- 💾 **Persistent Storage**: Training data is saved and persists across restarts

## Installation

### HACS (Recommended)

1. Open HACS in Home Assistant
2. Go to "Integrations" → Three dots menu → "Custom repositories"
3. Add this repository URL: `https://github.com/nilsreiter/ha-predictions`
4. Select category "Integration" and click "Add"
5. Search for "HA Predictions" and install

### Manual Installation

1. Copy the `custom_components/ha_predictions/` directory to your Home Assistant `config/custom_components/` folder
2. Restart Home Assistant

## Configuration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for **HA Predictions**
3. Select:
   - **Target Entity**: The entity to predict (e.g., light or switch)
   - **Feature Entities**: Entities to use as prediction features (e.g., sensors, time, other lights)

You can modify feature entities later via **Configure**, but this will reset your training data.

## Usage

The integration creates these entities:

- **Sensors**: Prediction Performance (accuracy %), Dataset Size (sample count),
  Current Prediction (state + confidence), Last Training (timestamp)
- **Buttons**: Store Instance (manual save), Evaluate Model / Train Production Model
  (requires 10+ samples)
- **Mode Selector**: TRAINING (collect data) / PRODUCTION (make predictions)

## Workflow

1. **Training Phase**: Set mode to TRAINING and let your home operate normally for days/weeks. Data is automatically collected. Monitor Dataset Size sensor.
2. **Evaluate Model**: Once you have 10+ samples, click **Evaluate Model**. Check
   the Prediction Performance sensor for accuracy.
3. **Production**: Set mode to PRODUCTION. The final model is trained automatically
   from all available data and restored automatically after restarting Home Assistant.
4. **Automation**: Create a Home Assistant automation that triggers when the prediction changes to control your target entity (e.g., switch lights). This is a security measure to ensure predictions don't directly control devices.

`Last Training` changes only after successful evaluation or production training. New
training events continue increasing `Dataset Size` without retraining the model. The
dataset sensor attributes `model_is_stale` and `samples_at_last_training` show whether newer
samples exist than those used by the current model.

Automatic dataset collection only runs in `TRAINING`. In `PRODUCTION`, state events
create predictions but are not added as training rows, preventing automations driven
by a prediction from feeding their own result back into the model. `Store Instance`
remains an explicit manual exception.

## Example Use Cases

- Predict presence/arrivals based on time and sensors
- Automate lighting based on motion, ambient light, and time
- Control climate based on occupancy and temperature
- Manage energy usage based on historical patterns

## Requirements

- Home Assistant 2025.2.4 or later
- Python packages: pandas, numpy (auto-installed)

## Development

Based on [integration_blueprint](https://github.com/ludeeus/integration_blueprint). See [CONTRIBUTING.md](CONTRIBUTING.md) for setup details.

Local Python dependencies and the project virtual environment are managed with
[uv](https://docs.astral.sh/uv/). Install uv, then run:

```bash
uv sync --dev
uv run pytest
```

`uv sync` creates and maintains `.venv` from the committed `uv.lock` file.

### Docker development environment

The repository includes a standalone Home Assistant development environment. It
runs Home Assistant in Docker and loads the integration directly from the local
working tree. The production integration is not installed through HACS in this
environment.

Docker and Docker Compose are required.

#### Start Home Assistant

From the repository root, start the development instance:

```bash
docker compose up -d
docker compose logs -f homeassistant
```

Open <http://localhost:8123> and complete the initial Home Assistant setup. The
Compose configuration uses these mounts:

- `./custom_components` → `/config/custom_components`: the current local source
  code of the integration
- `./config` → `/config`: the local Home Assistant configuration and runtime data

The local `config` directory is ignored by Git except for
`config/configuration.yaml`. Training data, the recorder database and Home
Assistant-generated files therefore stay local.

#### Configure the test integration

The development configuration creates these test entities:

| Purpose | Entity |
| --- | --- |
| Prediction target | `input_boolean.test_prediction_target` |
| Motion feature | `input_boolean.test_motion` |
| Presence feature | `input_boolean.test_presence` |
| Simulated hour | `input_number.test_hour` |
| Ambient light | `input_number.test_ambient_lux` |

In Home Assistant, go to **Settings → Devices & services → Add integration** and
add **HA Predictions**. Configure it as follows:

- **Target Entity:** `input_boolean.test_prediction_target`
- **Feature Entities:** `input_boolean.test_motion`, `input_boolean.test_presence`,
  `input_number.test_hour`, and `input_number.test_ambient_lux`

Changing the feature entities resets the training data. Keep the operation mode on
`TRAINING` while collecting data. For a first run, select `None` as the sampling
strategy so that the raw dataset can be inspected without resampling.

#### Generate realistic test data

The repository contains a reproducible test-data generator. It uses the Home
Assistant REST API to change the test entities, so the integration processes the
data exactly like normal state changes.

Create a Long-Lived Access Token in your Home Assistant user profile. Do not commit
the token or put it into a configuration file. Run the generator locally with:

```bash
uv run python scripts/generate_training_data.py --token YOUR_TOKEN
```

By default, it creates 180 simulated situations across several days. Each situation
varies the hour, presence, motion and ambient light. The target is derived from
these values with a small amount of noise to resemble real household data. Because
each situation changes several Home Assistant entities, the integration usually
collects several hundred dataset rows.

Useful options:

```bash
uv run python scripts/generate_training_data.py \
  --token YOUR_TOKEN \
  --samples 500 \
  --seed 123 \
  --delay 0.05
```

Use `--url` when Home Assistant is not running at `http://localhost:8123`. The
fixed `--seed` makes a test run reproducible.

For continuous observation in training or production mode, run the live simulator:

```bash
export HA_TOKEN=YOUR_TOKEN
uv run python scripts/simulate_test_entities.py
```

It advances a simulated clock by 15 minutes per iteration and updates the test
entities every two seconds. Presence, motion, daylight and weather evolve as a
coherent household scenario. The target follows those features with some hysteresis
and noise, so predictions have a learnable pattern without becoming perfect.

Useful live options:

```bash
uv run python scripts/simulate_test_entities.py \
  --interval 1 \
  --step-minutes 30 \
  --noise 0.08 \
  --iterations 200
```

An iteration count of `0` runs until `Ctrl+C`. Use `--dry-run` to inspect generated
states without sending API requests. `HA_URL` can be used instead of `--url`.

#### Train and test predictions

After generating data:

1. Check the **Dataset size** sensor.
2. Press **Evaluate Model**.
3. Check **Prediction Performance** and the per-class metrics in its attributes.
4. Change **Operation Mode** to `PRODUCTION`; final training starts automatically.
5. Change the test feature entities and observe **Current Prediction**.

Use **Rebuild Dataset from History** to replace the CSV with recorder data. The
rebuild uses the Operation Mode history and excludes timestamps where the integration
was in `PRODUCTION`; state changes from those periods are still used to reconstruct
the first complete state after returning to `TRAINING`. In production mode, the
cleaned dataset is retrained automatically. Recorder rebuilds are unavailable for
attribute targets because their historical attributes are not imported yet.

Dataset filters are configured from the integration's **Configure** dialog:

- inclusive start and end dates
- a daily time range, including ranges across midnight
- selected weekdays
- a minimum interval in seconds between accepted rows
- whether every feature must have a valid state
- whether `PRODUCTION` rows may be included

Production rows are excluded by default because including them can feed automation
effects caused by a prediction back into the model. Saving filter options does not
replace the existing CSV. The dataset sensor reports `rebuild_required` until
**Rebuild Dataset from History** applies the current filters. `Store Instance` is an
explicit exception and stores the current state regardless of automatic filters.

#### Fast development cycle

Python modules are loaded when Home Assistant starts. After changing integration
code, restart the container to load the new code:

```bash
docker compose restart homeassistant
docker compose logs -f homeassistant
```

Run the standalone tests and code checks from the repository root:

```bash
scripts/test
scripts/lint
```

The existing devcontainer remains available as an alternative. It runs the same
Home Assistant configuration with `scripts/develop`.

#### Stop or reset the development instance

Stop the instance with:

```bash
docker compose down
```

To reset the local Home Assistant state, remove the generated files below `config/`
while keeping `config/configuration.yaml`. This removes the local HA user, recorder
history, integration entries and training data.

## Support

- [Report bugs and request features](https://github.com/nilsreiter/ha-predictions/issues)
- [Documentation](https://github.com/nilsreiter/ha-predictions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- Created by [@nilsreiter](https://github.com/nilsreiter)
- Based on [integration_blueprint](https://github.com/ludeeus/integration_blueprint)

---

[releases-shield]: https://img.shields.io/github/release/nilsreiter/ha-predictions.svg?style=for-the-badge
[releases]: https://github.com/nilsreiter/ha-predictions/releases
[commits-shield]: https://img.shields.io/github/commit-activity/y/nilsreiter/ha-predictions.svg?style=for-the-badge
[commits]: https://github.com/nilsreiter/ha-predictions/commits/main
[license-shield]: https://img.shields.io/github/license/nilsreiter/ha-predictions.svg?style=for-the-badge
[hacs]: https://github.com/hacs/integration
[hacsbadge]: https://img.shields.io/badge/HACS-Custom-orange.svg?style=for-the-badge
