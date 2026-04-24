const ASIAN_COUNTRY_CODES = new Set([
  'AF','AM','AZ','BH','BD','BT','BN','KH','CN','CY','GE','IN','ID',
  'IR','IQ','IL','JP','JO','KZ','KW','KG','LA','LB','MY','MV','MN',
  'MM','NP','OM','PK','PH','QA','RU','SA','SG','LK','SY','TJ','TH',
  'TR','TM','AE','UZ','VN','YE'
]);

export default async function handler(req, res) {
  const query = req.query.q;
  if (!query || query.length < 2) return res.json([]);

  if (!process.env.OPENWEATHER_API_KEY) {
    return res.status(503).json({ error: 'OPENWEATHER_API_KEY not configured on server' });
  }

  try {
    const url = `http://api.openweathermap.org/geo/1.0/direct?q=${encodeURIComponent(query)}&limit=10&appid=${process.env.OPENWEATHER_API_KEY}`;
    const response = await fetch(url);
    const data = await response.json();

    const cities = data
      .filter(city => ASIAN_COUNTRY_CODES.has(city.country))
      .map(city => ({
        label: `${city.name}${city.state ? `, ${city.state}` : ''}, ${city.country}`,
        value: `${city.lat},${city.lon}`,
        lat: city.lat,
        lon: city.lon,
      }));

    res.json(cities);
  } catch (error) {
    res.status(500).json({ error: 'Failed to search cities' });
  }
}
