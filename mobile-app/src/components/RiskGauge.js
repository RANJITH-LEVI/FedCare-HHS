import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, RISK_LEVELS } from '../constants/theme';

export default function RiskGauge({ probability, riskLevel, confidence }) {
  const pct = Math.round((probability || 0) * 100);
  const confPct = Math.round((confidence || 0.85) * 100);
  
  const levelInfo = RISK_LEVELS[riskLevel] || RISK_LEVELS['Low Risk'];

  return (
    <View style={[styles.container, { borderColor: levelInfo.border, backgroundColor: levelInfo.bg }]}>
      <View style={styles.topRow}>
        <View style={styles.badgeRow}>
          <Ionicons name={levelInfo.icon} size={18} color={levelInfo.color} />
          <Text style={[styles.badgeText, { color: levelInfo.color }]}>
            {levelInfo.badge}
          </Text>
        </View>
        <View style={styles.confBadge}>
          <Text style={styles.confText}>{confPct}% Confidence</Text>
        </View>
      </View>

      <View style={styles.scoreSection}>
        <Text style={[styles.percentNumber, { color: levelInfo.color }]}>{pct}%</Text>
        <Text style={styles.percentLabel}>Cardiovascular Disease Probability</Text>
      </View>

      {/* Progress Track */}
      <View style={styles.trackBackground}>
        <View
          style={[
            styles.trackFill,
            {
              width: `${Math.min(Math.max(pct, 4), 100)}%`,
              backgroundColor: levelInfo.color,
            },
          ]}
        />
      </View>

      <View style={styles.thresholdRow}>
        <Text style={styles.thresholdText}>0% Low</Text>
        <Text style={styles.thresholdText}>50% Borderline</Text>
        <Text style={styles.thresholdText}>100% Critical</Text>
      </View>

      {/* Clinical Recommendation */}
      <View style={styles.recommendationCard}>
        <Ionicons name="medical" size={16} color={levelInfo.color} style={{ marginTop: 2 }} />
        <Text style={styles.recommendationText}>{levelInfo.recommendation}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 16,
    borderWidth: 1.5,
    padding: 18,
    marginVertical: 12,
  },
  topRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  badgeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  badgeText: {
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  confBadge: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },
  confText: {
    color: COLORS.textSecondary,
    fontSize: 11,
    fontWeight: '600',
  },
  scoreSection: {
    alignItems: 'center',
    marginVertical: 10,
  },
  percentNumber: {
    fontSize: 48,
    fontWeight: '900',
    letterSpacing: -1,
  },
  percentLabel: {
    fontSize: 13,
    color: COLORS.textSecondary,
    fontWeight: '500',
    marginTop: 2,
  },
  trackBackground: {
    height: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: 5,
    overflow: 'hidden',
    marginTop: 10,
  },
  trackFill: {
    height: '100%',
    borderRadius: 5,
  },
  thresholdRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 6,
    marginBottom: 14,
  },
  thresholdText: {
    fontSize: 10,
    color: COLORS.textMuted,
    fontWeight: '500',
  },
  recommendationCard: {
    flexDirection: 'row',
    gap: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.25)',
    borderRadius: 10,
    padding: 12,
  },
  recommendationText: {
    flex: 1,
    fontSize: 12,
    color: COLORS.text,
    lineHeight: 18,
  },
});
