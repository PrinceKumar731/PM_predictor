export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  if (!process.env.OPENWEATHER_API_KEY) {
    return res.status(503).json({ error: 'OPENWEATHER_API_KEY not configured on server' });
  }

  const { latitude, longitude, year, month } = req.body;

  try {
    const url = `http://api.openweathermap.org/data/2.5/air_pollution?lat=${latitude}&lon=${longitude}&appid=${process.env.OPENWEATHER_API_KEY}`;
    const response = await fetch(url);
    const data = await response.json();

    if (!data.list || data.list.length === 0) {
      throw new Error('No data received from OpenWeather');
    }

    const components = data.list[0].components;

    res.json({
      latitude,
      longitude,
      year,
      month,
      predicted_pm25: components.pm2_5,
      nearest_grid_latitude: latitude,
      nearest_grid_longitude: longitude,
      top_factors: [
        { feature: 'CO (Carbon Monoxide)', value: components.co, effect: 'increased' },
        { feature: 'NO2 (Nitrogen Dioxide)', value: components.no2, effect: 'increased' },
        { feature: 'O3 (Ozone)', value: components.o3, effect: 'decreased' },
        { feature: 'SO2 (Sulphur Dioxide)', value: components.so2, effect: 'increased' },
        { feature: 'PM10', value: components.pm10, effect: 'increased' },
      ],
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch live data', detail: error.message });
  }
}
