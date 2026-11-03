export function getScoreColor(score) {
  if (score == null) return '#9ca3af';
  if (score >= 70) return '#7c3aed';
  if (score >= 55) return '#0b8a3e';
  if (score >= 45) return '#d9840d';
  return '#c92a2a';
}

export function getScoreLabel(score) {
  if (score == null) return 'N/A';
  if (score >= 70) return 'Excellent';
  if (score >= 55) return 'Good';
  if (score >= 45) return 'Fair';
  if (score >= 30) return 'Weak';
  return 'Poor';
}

export function getScoreBg(score) {
  if (score == null) return '#f3f4f6';
  if (score >= 70) return '#ede9fe';
  if (score >= 55) return '#d3f9d8';
  if (score >= 45) return '#fff3bf';
  return '#ffe3e3';
}

export function getScoreGradient(score) {
  if (score == null) return 'linear-gradient(135deg, #9ca3af, #6b7280)';
  if (score >= 70) return 'linear-gradient(135deg, #7c3aed, #8b5cf6)';
  if (score >= 55) return 'linear-gradient(135deg, #0b8a3e, #2b9a3e)';
  if (score >= 45) return 'linear-gradient(135deg, #d9840d, #e09800)';
  return 'linear-gradient(135deg, #c92a2a, #e03131)';
}
