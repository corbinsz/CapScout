/**
 * Answers the same calls as the Python API from a saved Austin file.
 * Used when VITE_API_URL is unset, which is the Vercel preview.
 */
import { MARKETS } from '../utils/markets';
import { scoreProperty } from '../utils/scoringEngine';

let bundlePromise = null;

function loadBundle() {
  if (!bundlePromise) {
    bundlePromise = fetch('/demo/austin.json').then((response) => {
      if (!response.ok) throw new Error('The Austin sample did not load.');
      return response.json();
    });
  }
  return bundlePromise;
}

function ok(data) {
  return Promise.resolve({ data, status: 200 });
}

/** @param {string} message @param {number} status */
function fail(message, status) {
  const error = /** @type {Error & { response: { status: number, data: { error: string, upgrade_required?: boolean } } }} */ (new Error(message));
  error.response = { status, data: { error: message, upgrade_required: status === 402 } };
  return error;
}

function matches(item, params) {
  const price = item.price || 0;
  if (params.min_price != null && params.min_price !== '' && price < Number(params.min_price)) return false;
  if (params.max_price != null && params.max_price !== '' && price > Number(params.max_price)) return false;
  if (params.bedrooms != null && params.bedrooms !== '' && item.bedrooms !== Number(params.bedrooms)) return false;
  if (params.property_type && !(item.property_type || '').toLowerCase().includes(String(params.property_type).toLowerCase())) return false;
  if (params.str_viable === true || params.str_viable === 'true') {
    if (item.confidence_level === 'low') return false;
  }
  return true;
}

function sortItems(items, params) {
  const desc = (params.sort_order || 'desc') !== 'asc';
  const key = params.sort_by === 'price'
    ? (item) => item.price || 0
    : (item) => (item.score == null ? -1 : item.score);
  return [...items].sort((a, b) => (desc ? key(b) - key(a) : key(a) - key(b)));
}

async function properties(params) {
  const bundle = await loadBundle();
  const city = params.city || 'austin';
  if (city !== 'austin') {
    return {
      items: [],
      total: 0,
      total_available: 0,
      page: 1,
      per_page: Number(params.per_page) || 20,
      data_source: 'demo',
      data_source_label: 'Austin sample',
      data_licensed: false,
      message: `${city} is not in this preview. Austin is the saved sample you can click through. Other cities appear when the live server is connected.`,
    };
  }
  const filtered = sortItems(bundle.listings.filter((item) => matches(item, params)), params);
  const page = Number(params.page) || 1;
  const perPage = Number(params.per_page) || 20;
  const start = (page - 1) * perPage;
  return {
    items: filtered.slice(start, start + perPage),
    total: filtered.length,
    total_available: filtered.length,
    page,
    per_page: perPage,
    data_source: 'demo',
    data_source_label: 'Austin sample',
    data_licensed: false,
    message: bundle.message,
  };
}

async function propertyDetail(id) {
  const bundle = await loadBundle();
  const item = bundle.listings.find((row) => row.id === id);
  if (!item) throw fail('That home is not in the Austin sample.', 404);
  return { property: item, comparables: [], neighborhood_stats: null, demo: true };
}

async function predict(body) {
  const bundle = await loadBundle();
  const beds = Number(body.bedrooms) || 3;
  const lat = Number(body.latitude) || 30.27;
  const lng = Number(body.longitude) || -97.74;
  const ranked = [...bundle.listings].sort((a, b) => {
    const da = Math.abs((a.bedrooms || 0) - beds) * 5 + Math.abs((a.latitude || 0) - lat) + Math.abs((a.longitude || 0) - lng);
    const db = Math.abs((b.bedrooms || 0) - beds) * 5 + Math.abs((b.latitude || 0) - lat) + Math.abs((b.longitude || 0) - lng);
    return da - db;
  });
  const near = ranked[0];
  const nightly = near?.predicted?.predicted_nightly_rate || 180;
  const occupancy = near?.predicted?.predicted_occupancy_rate || 0.35;
  const annual = nightly * occupancy * 365;
  const assumptions = bundle.assumptions;
  const scored = scoreProperty({
    purchasePrice: Number(body.purchase_price) || 0,
    annualRevenue: annual,
    occupancy,
    mortgageRate: 0.07,
    mortgageTerm: 30,
    downPaymentPct: 0.2,
    taxRate: assumptions.taxRate,
    insuranceRate: assumptions.insuranceRate,
    hoaAnnual: 0,
    strPermitAnnual: assumptions.strPermitAnnual,
    cityExpenses: assumptions,
  });
  const fin = scored.financials;
  return {
    score: scored.score,
    explanation: `A nearby ${beds}-bedroom sample rents for about $${Math.round(nightly)} a night. At this price the cap rate is ${fin.capRate}% and cash flow is $${fin.monthlyCashflow} a month. This is the Austin preview, not a live model call, and it is not financial advice.`,
    prediction: {
      predicted_nightly_rate: nightly,
      predicted_occupancy_rate: occupancy,
      predicted_monthly_revenue: Math.round(annual / 12),
      predicted_annual_revenue: Math.round(annual),
    },
    financials: {
      cap_rate: fin.capRate,
      roi: fin.roi,
      dscr: fin.dscr,
      monthly_cashflow: fin.monthlyCashflow,
      annual_cashflow: fin.annualCashflow,
      noi: fin.noi,
    },
  };
}

function cities() {
  return MARKETS.map((market) => ({
    key: market.key,
    name: market.name,
    state: market.state,
    display: `${market.name}${market.state ? `, ${market.state}` : ''}`,
    latitude: market.latitude,
    longitude: market.longitude,
    default_zoom: 11,
    has_data: market.key === 'austin',
    has_models: market.key === 'austin',
  }));
}

async function handle(method, url, config = {}) {
  const path = String(url || '').split('?')[0];
  const params = config.params || {};
  if (path === '/auth/me') return ok({ user: null });
  if (path === '/cities') return ok(cities());
  if (path === '/properties') return ok(await properties(params));
  if (path.startsWith('/properties/') && path.endsWith('/feature-importance')) {
    return ok({
      price: { label: 'Price drivers', features: [{ name: 'Location', importance: 0.4 }, { name: 'Bedrooms', importance: 0.2 }] },
      occupancy: { label: 'Occupancy drivers', features: [{ name: 'Location', importance: 0.5 }] },
    });
  }
  if (path.startsWith('/properties/')) {
    const id = decodeURIComponent(path.slice('/properties/'.length));
    return ok(await propertyDetail(id));
  }
  if (path === '/model-metrics') {
    const bundle = await loadBundle();
    return ok(bundle.metrics || { available: false });
  }
  if (path === '/analytics') {
    const bundle = await loadBundle();
    const sample = bundle.listings.slice(0, 200);
    const avg = sample.reduce((sum, item) => sum + (item.predicted?.predicted_nightly_rate || 0), 0) / (sample.length || 1);
    return ok({
      total_listings: bundle.listings.length,
      avg_nightly_rate: Math.round(avg),
      median_nightly_rate: Math.round(avg),
      avg_occupancy: 0.3,
      neighborhoods: [],
      bedroom_breakdown: [],
      price_distribution: {},
      message: bundle.message,
    });
  }
  if (path === '/markets/leaders') {
    const bundle = await loadBundle();
    const best = [...bundle.listings].sort((a, b) => (b.score || 0) - (a.score || 0))[0];
    return ok({
      note: 'This preview only has the Austin sample.',
      items: [{
        city: 'austin',
        display: 'Austin, TX',
        top_score: best?.score,
        address: best?.address,
        property_id: best?.id,
        price: best?.price,
        monthly_cashflow: best?.financials?.monthly_cashflow,
        homes_scored: bundle.listings.length,
      }],
    });
  }
  if (path === '/search') {
    const bundle = await loadBundle();
    const q = String(params.q || '').toLowerCase();
    const results = bundle.listings
      .filter((item) => (item.address || '').toLowerCase().includes(q))
      .slice(0, 8)
      .map((item) => ({
        id: item.id,
        name: item.address,
        address: item.address,
        price: item.price,
        bedrooms: item.bedrooms,
        bathrooms: item.bathrooms,
        source: 'demo',
      }));
    return ok({ results });
  }
  if (path === '/settings/keys') {
    return ok({
      rentcast: false,
      redfin_fallback_enabled: false,
      listing_source: 'demo',
      listing_label: 'Saved Austin sample',
      listing_licensed: false,
      key_write_enabled: false,
    });
  }
  if (path === '/billing/status') {
    return ok({ stripe_configured: false, price_label: '$29/month', tier: null, signed_in: false });
  }
  if (method === 'post' && path === '/predict') return ok(await predict(config.data || {}));
  if (method === 'post' && path === '/alerts') {
    return ok({
      ok: true,
      delivery: 'preview',
      message: 'This preview does not send email. Your note stayed in the browser. Nothing was sent.',
    });
  }
  if (method === 'post' && (path === '/auth/register' || path === '/auth/login')) {
    throw fail('Accounts turn on when the live server is connected. This preview needs no sign-in.', 400);
  }
  if (method === 'post' && path === '/auth/logout') return ok({ ok: true });
  if (method === 'post' && path === '/billing/checkout') {
    return ok({
      mode: 'stub',
      configured: false,
      message: 'Checkout is not connected on this preview. No card is charged.',
    });
  }
  if (path === '/reports/top-deals') {
    throw fail('The downloadable report needs the live server. You can still read the deals on this page.', 402);
  }
  throw fail('That action is not in the preview.', 404);
}

const demoApi = {
  get(url, config) {
    return handle('get', url, config);
  },
  post(url, data, config) {
    return handle('post', url, { ...config, data });
  },
  interceptors: { response: { use() {} } },
};

export default demoApi;
