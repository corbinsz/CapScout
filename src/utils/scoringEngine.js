/**
 * Client-side port of backend/services/scoring_service.py
 * Used for sensitivity analysis -- no API calls needed.
 *
 * Itemized expense model: platform fees, cleaning per turnover,
 * supplies, utilities (occupancy-driven), maintenance, fixed costs.
 */

// Constants matching backend
const PLATFORM_FEE_RATE = 0.03;
const SUPPLIES_PER_TURNOVER = 25;
const MAINTENANCE_RATE = 0.01;
const WIFI_MONTHLY = 75;
const MISC_MONTHLY = 50;

function calculateAnnualMortgage(loanAmount, rate = 0.07, years = 30) {
  const monthlyRate = rate / 12;
  const n = years * 12;
  if (monthlyRate === 0) return (loanAmount / n) * 12;
  const monthly = loanAmount * (monthlyRate * Math.pow(1 + monthlyRate, n)) / (Math.pow(1 + monthlyRate, n) - 1);
  return monthly * 12;
}

function calculateOperatingExpenses(grossRevenue, occupancy, purchasePrice, cityExpenses) {
  const cleaningCost = cityExpenses.cleaningPerTurnover || 120;
  const avgStay = cityExpenses.avgStayNights || 3.2;
  const baseUtil = cityExpenses.baseUtilitiesMonthly || 250;

  const occupiedNights = occupancy * 365;
  const turnovers = avgStay > 0 ? occupiedNights / avgStay : 0;

  const platformFees = grossRevenue * PLATFORM_FEE_RATE;
  const cleaning = turnovers * cleaningCost;
  const supplies = turnovers * SUPPLIES_PER_TURNOVER;
  const utilities = baseUtil * 12 * (0.6 + 0.4 * occupancy);
  const fixedCosts = (WIFI_MONTHLY + MISC_MONTHLY) * 12;
  const maintenance = purchasePrice * MAINTENANCE_RATE;

  return platformFees + cleaning + supplies + utilities + fixedCosts + maintenance;
}

/**
 * Recalculate score and financials with adjusted inputs.
 *
 * @param {object} params
 * @param {number} params.purchasePrice
 * @param {number} params.annualRevenue
 * @param {number} params.occupancy - decimal (0-1)
 * @param {number} params.mortgageRate - decimal (e.g. 0.07)
 * @param {number} params.mortgageTerm - years
 * @param {number} params.downPaymentPct - decimal (e.g. 0.20)
 * @param {number} params.taxRate - decimal
 * @param {number} params.insuranceRate - decimal
 * @param {number} params.hoaAnnual
 * @param {number} params.strPermitAnnual
 * @param {object} params.cityExpenses - { cleaningPerTurnover, avgStayNights, baseUtilitiesMonthly }
 */
export function scoreProperty({
  purchasePrice,
  annualRevenue,
  occupancy,
  mortgageRate = 0.07,
  mortgageTerm = 30,
  downPaymentPct = 0.20,
  taxRate = 0.018,
  insuranceRate = 0.005,
  hoaAnnual = 0,
  strPermitAnnual = 0,
  cityExpenses = {},
}) {
  if (purchasePrice <= 0) return { score: 0, financials: {} };

  const grossRevenue = annualRevenue;
  const operatingExpenses = calculateOperatingExpenses(grossRevenue, occupancy, purchasePrice, cityExpenses);
  const noi = grossRevenue - operatingExpenses;

  const downPayment = purchasePrice * downPaymentPct;
  const loanAmount = purchasePrice - downPayment;
  const annualMortgage = calculateAnnualMortgage(loanAmount, mortgageRate, mortgageTerm);
  const annualTaxes = purchasePrice * taxRate;
  const annualInsurance = purchasePrice * insuranceRate;

  const totalDebtService = annualMortgage + annualTaxes + annualInsurance + strPermitAnnual + hoaAnnual;
  const annualCashflow = noi - totalDebtService;

  const grossYield = grossRevenue / purchasePrice;
  const capRate = noi / purchasePrice;
  const roi = downPayment > 0 ? annualCashflow / downPayment : 0;
  const dscr = totalDebtService > 0 ? noi / totalDebtService : 0;

  // Scoring components (same thresholds as backend)
  const yieldScore = Math.min(25, Math.max(0, (grossYield / 0.20) * 25));

  let roiScore;
  if (roi >= 0) {
    roiScore = Math.min(25, 5 + (roi / 0.20) * 20);
  } else {
    roiScore = Math.max(0, 5 + (roi / 0.10) * 5);
  }

  const capRateScore = Math.min(20, Math.max(0, (capRate / 0.10) * 20));

  let dscrScore;
  if (dscr >= 1.0) {
    dscrScore = Math.min(20, 10 + ((dscr - 1.0) / 0.5) * 10);
  } else {
    dscrScore = Math.max(0, (dscr / 1.0) * 10);
  }

  const occScore = Math.min(10, Math.max(0, (occupancy / 0.75) * 10));

  const score = Math.round(Math.min(100, Math.max(0, yieldScore + roiScore + capRateScore + dscrScore + occScore)) * 10) / 10;

  return {
    score,
    financials: {
      annualCashflow: Math.round(annualCashflow),
      monthlyCashflow: Math.round(annualCashflow / 12),
      capRate: Math.round(capRate * 10000) / 100,
      roi: Math.round(roi * 10000) / 100,
      grossYield: Math.round(grossYield * 10000) / 100,
      dscr: Math.round(dscr * 100) / 100,
      noi: Math.round(noi),
    },
  };
}
