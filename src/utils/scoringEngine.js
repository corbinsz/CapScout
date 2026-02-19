/**
 * Client-side port of backend/services/scoring_service.py
 * Used for sensitivity analysis — no API calls needed.
 */

function calculateAnnualMortgage(loanAmount, rate = 0.07, years = 30) {
  const monthlyRate = rate / 12;
  const n = years * 12;
  if (monthlyRate === 0) return (loanAmount / n) * 12;
  const monthly = loanAmount * (monthlyRate * Math.pow(1 + monthlyRate, n)) / (Math.pow(1 + monthlyRate, n) - 1);
  return monthly * 12;
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
 * @param {number} params.expenseRatio - decimal
 * @param {number} params.hoaAnnual
 * @param {number} params.strPermitAnnual
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
  expenseRatio = 0.35,
  hoaAnnual = 0,
  strPermitAnnual = 0,
}) {
  if (purchasePrice <= 0) return { score: 0, financials: {} };

  const grossRevenue = annualRevenue;
  const operatingExpenses = grossRevenue * expenseRatio;
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
