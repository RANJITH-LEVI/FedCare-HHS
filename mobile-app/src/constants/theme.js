export const COLORS = {
  background: '#0B1120',
  surface: '#1E293B',
  surfaceLight: '#334155',
  surfaceBorder: '#475569',
  
  primary: '#0284C7',       // Medical Sky Blue
  primaryLight: '#38BDF8',
  primaryGlow: 'rgba(56, 189, 248, 0.25)',
  
  accent: '#6366F1',        // Indigo
  
  riskLow: '#10B981',       // Emerald
  riskLowBg: 'rgba(16, 185, 129, 0.15)',
  riskLowBorder: '#059669',
  
  riskModerate: '#F59E0B',  // Amber
  riskModerateBg: 'rgba(245, 158, 11, 0.15)',
  riskModerateBorder: '#D97706',
  
  riskHigh: '#EF4444',      // Crimson
  riskHighBg: 'rgba(239, 68, 68, 0.15)',
  riskHighBorder: '#DC2626',
  
  text: '#F8FAFC',
  textSecondary: '#94A3B8',
  textMuted: '#64748B',
  
  cardBg: '#131D31',
  inputBg: '#0F172A',
  badgeBg: '#1E293B',
  
  white: '#FFFFFF',
  danger: '#F43F5E',
  success: '#10B981',
};

export const RISK_LEVELS = {
  'Low Risk': {
    color: COLORS.riskLow,
    bg: COLORS.riskLowBg,
    border: COLORS.riskLowBorder,
    badge: 'HEALTHY / LOW RISK',
    icon: 'shield-checkmark',
    recommendation: 'Annual cardiovascular checkup recommended. Maintain regular aerobic exercise and Mediterranean-style diet.'
  },
  'Moderate Risk': {
    color: COLORS.riskModerate,
    bg: COLORS.riskModerateBg,
    border: COLORS.riskModerateBorder,
    badge: 'ELEVATED RISK',
    icon: 'warning',
    recommendation: 'Consult cardiologist for confirmatory stress ECG and lipid profiling. Monitor blood pressure weekly.'
  },
  'High Risk': {
    color: COLORS.riskHigh,
    bg: COLORS.riskHighBg,
    border: COLORS.riskHighBorder,
    badge: 'HIGH CARDIOVASCULAR RISK',
    icon: 'alert-circle',
    recommendation: 'Urgent cardiology evaluation warranted. Consider coronary CT angiography and immediate risk factor intervention.'
  }
};
