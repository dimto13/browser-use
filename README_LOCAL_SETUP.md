
# 🛠️ Local Setup for Development

If you want to run this repository locally with a custom environment:

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv-browse-dim
    ```

2.  **Install dependencies:**
    ```bash
    ./venv-browse-dim/bin/pip install -e .
    ```

3.  **Install Playwright browsers:**
    ```bash
    ./venv-browse-dim/bin/playwright install
    ```

4.  **Run your script:**
    ```bash
    ./venv-browse-dim/bin/python main/run_browser_with_ollama.py
    ```

**Note:** Ensure you have Ollama running with the required model (e.g., `deepseek-r1:32b`) before running the script.
