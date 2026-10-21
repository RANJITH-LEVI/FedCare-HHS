import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { COLORS } from '../constants/theme';

const FEATURE_LABELS = {
  age: 'Age',
  sex: 'Biological Sex',
  cp: 'Chest Pain Type',
  trestbps: 'Resting Blood Pressure',
  chol: 'Serum Cholesterol',
  fbs: 'Fasting Blood Sugar',
  restecg: 'Resting ECG Findings',
  thalach: 'Max Heart Rate',
  exang: 'Exercise Angina',
  oldpeak: 'ST Depression',
  slope: 'Peak Exercise ST Slope',
  ca: 'Major Vessels (Fluoroscopy)',
  thal: 'Thalassemia Defect',
};

export default function ShapBarChart({ shapValues = {}, patientData = {} }) {
  // Convert shapValues dictionary to sorted array by absolute magnitude
  const items = Object.entries(shapValues)
    .map(([feature, val]) => ({
      feature,
      label: FEATURE_LABELS[feature] || feature.toUpperCase(),
      val: parseFloat(val) || 0,
      absVal: Math.abs(parseFloat(val) || 0),
      patientVal: patientData[feature],
    }))
    .sort((a, b) => b.absVal - a.absVal)
    .slice(0, 8); // Top 8 contributors

  const maxVal = Math.max(...items.map(i => i.absVal), 0.05);

  return (
    <View style={styles.container}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>SHAP Clinical Risk Attribution</Text>
        <Text style={styles.subtitle}>Key Drivers of Model Decision</Text>
      </View>

      <View style={styles.legendRow}>
        <View style={styles.legendItem}>
          <View style={[styles.legendSquare, { backgroundColor: COLORS.riskHigh }]} />
          <Text style={styles.legendText}>Elevates Risk (+)</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendSquare, { backgroundColor: COLORS.riskLow }]} />
          <Text style={styles.legendText}>Protective (-)</Text>
        </View>
      </View>

      <View style={styles.chartList}>
        {items.map((item, index) => {
          const isRisk = item.val >= 0;
          const barColor = isRisk ? COLORS.riskHigh : COLORS.riskLow;
          const barWidthPercent = Math.min((item.absVal / maxVal) * 100, 100);

          return (
            <View key={item.feature} style={styles.barItem}>
              <View style={styles.barLabelRow}>
                <Text style={styles.featureName} numberOfLines={1}>
                  {item.label}
                </Text>
                <Text style={styles.patientValue}>
                  {item.patientVal !== undefined ? `val: ${item.patientVal}` : ''}
                </Text>
              </View>

              <View style={styles.track}>
                <View
                  style={[
                    styles.fill,
                    {
                      width: `${barWidthPercent}%`,
                      backgroundColor: barColor,
                    },
                  ]}
                />
                <Text style={[styles.valText, { color: barColor }]}>
                  {isRisk ? `+${item.val.toFixed(3)}` : item.val.toFixed(3)}
                </Text>
              </View>
            </View>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    marginTop: 12,
  },
  headerRow: {
    marginBottom: 8,
  },
  title: {
    fontSize: 15,
    fontWeight: '700',
    color: COLORS.text,
  },
  subtitle: {
    fontSize: 12,
    color: COLORS.textSecondary,
    marginTop: 2,
  },
  legendRow: {
    flexDirection: 'row',
    gap: 16,
    marginVertical: 10,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceBorder,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendSquare: {
    width: 10,
    height: 10,
    borderRadius: 2,
  },
  legendText: {
    fontSize: 11,
    color: COLORS.textSecondary,
  },
  chartList: {
    gap: 12,
    marginTop: 4,
  },
  barItem: {},
  barLabelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  featureName: {
    fontSize: 12,
    fontWeight: '600',
    color: COLORS.text,
    flex: 1,
  },
  patientValue: {
    fontSize: 11,
    color: COLORS.textMuted,
  },
  track: {
    height: 14,
    backgroundColor: COLORS.surface,
    borderRadius: 7,
    flexDirection: 'row',
    alignItems: 'center',
    overflow: 'hidden',
    position: 'relative',
  },
  fill: {
    height: '100%',
    borderRadius: 7,
  },
  valText: {
    position: 'absolute',
    right: 8,
    fontSize: 10,
    fontWeight: '700',
  },
});
