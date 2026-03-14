import express from 'express';
import cors from 'cors';
import { exec } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = 8000;

app.use(cors());
app.use(express.json());

app.post('/api/predict', (req, res) => {
  const { latitude, longitude, year, month } = req.body;
  
  // Use the Python virtual env and run the command
  const pythonPath = path.join(__dirname, '../ml-service/.venv/bin/python');
  const mlServicePath = path.join(__dirname, '../ml-service');
  const command = `${pythonPath} -m src.predict --year ${year} --month ${month} --lat ${latitude} --lon ${longitude}`;

  console.log(`Executing: ${command}`);

  exec(command, { cwd: mlServicePath }, (error, stdout, stderr) => {
    if (error) {
      console.error(`Exec Error: ${error.message}`);
      return res.status(500).json({ error: 'Failed to run prediction', detail: stderr });
    }

    // Basic parser for the stdout
    const result = {
      latitude,
      longitude,
      year,
      month,
      predicted_pm25: null,
      top_factors: []
    };

    const lines = stdout.split('\n');
    lines.forEach(line => {
      if (line.includes('Predicted PM2.5:')) {
        result.predicted_pm25 = parseFloat(line.split(':')[1].trim());
      }
      if (line.includes('Nearest grid cell:')) {
        const parts = line.split(':')[1].split(',');
        result.nearest_grid_latitude = parseFloat(parts[0].split('=')[1]);
        result.nearest_grid_longitude = parseFloat(parts[1].split('=')[1]);
      }
      if (line.startsWith('- ')) {
        const parts = line.substring(2).split(':');
        const feature = parts[0].trim();
        const valueStr = parts[1].split(',')[0].split('=')[1];
        const effect = line.includes('increased') ? 'increased' : 'decreased';
        result.top_factors.push({ feature, value: parseFloat(valueStr), effect });
      }
    });

    res.json(result);
  });
});

app.listen(PORT, () => {
  console.log(`Node backend running at http://localhost:${PORT}`);
});
