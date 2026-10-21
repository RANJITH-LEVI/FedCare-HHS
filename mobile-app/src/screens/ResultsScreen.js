import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../constants/theme';
import RiskGauge from '../components/RiskGauge';
import ShapBarChart from '../components/ShapBarChart';
import storageService from '../services/storageService';

export default function ResultsScreen({ result, onNewAssessment, onSaved }) {
  const [saved, setSaved] = useState(false);

  if (!result) {
    return (
      <View style={styles.emptyContainer}>
        <Ionicons name="heart-outline" size={48} color={COLORS.textMuted} />
        <Text style={styles.emptyTitle}>No Assessment Completed</Text>
        <Text style={styles.emptySubtitle}>Run an evaluation from the Assessment tab first.</Text>
        <TouchableOpacity style={styles.startBtn} onPress={onNewAssessment}>
          <Text style={styles.startBtnText}>Start New Assessment</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const {
    patientData = {},
    probability = 0,
    riskLevel = 'Low Risk',
    confidence = 0.85,
    shapValues = {},
    narrative = '',
    scanned = false,
  } = result;

  const handleSave = async () => {
    try {
      await storageService.saveAssessment({
        patientData,
        probability,
        riskLevel,
        confidence,
        shapValues,
        scanned,
      });
      setSaved(true);
      if (onSaved) onSaved();
      Alert.alert('Saved to History', 'Patient cardiovascular evaluation saved locally.');
    } catch (e) {
      Alert.alert('Save Failed', e.message);
    }
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {/* Results Header */}
        <View style={styles.headerRow}>
          <View>
            <Text style={styles.headerTitle}>Cardiovascular Assessment</Text>
            <Text style={styles.headerSub}>
              {scanned ? 'Report Scanned with Gemini Vision' : 'Manual Intake Form'}
            </Text>
          </View>

          <TouchableOpacity
            style={[styles.saveBtn, saved && styles.saveBtnSaved]}
            onPress={handleSave}
            disabled={saved}
            activeOpacity={0.7}
          >
            <Ionicons
              name={saved ? 'checkmark-circle' : 'bookmark-outline'}
              size={18}
              color={saved ? COLORS.riskLow : COLORS.text}
            />
            <Text style={[styles.saveBtnText, saved && { color: COLORS.riskLow }]}>
              {saved ? 'Saved' : 'Save'}
            </Text>
          </TouchableOpacity>
        </View>

        {/* Visual Risk Gauge */}
        <RiskGauge probability={probability} riskLevel={riskLevel} confidence={confidence} />

        {/* Patient Key Metrics Strip */}
        <View style={styles.metricsStrip}>
          <View style={styles.metricItem}>
            <Text style={styles.metricLbl}>AGE/SEX</Text>
            <Text style={styles.metricVal}>
              {patientData.age} {patientData.sex === 1 ? 'M' : 'F'}
            </Text>
          </View>
          <View style={styles.metricDivider} />
          <View style={styles.metricItem}>
            <Text style={styles.metricLbl}>BLOOD PRESS.</Text>
            <Text style={styles.metricVal}>{patientData.trestbps} mmHg</Text>
          </View>
          <View style={styles.metricDivider} />
          <View style={styles.metricItem}>
            <Text style={styles.metricLbl}>CHOLESTEROL</Text>
            <Text style={styles.metricVal}>{patientData.chol} mg/dl</Text>
          </View>
          <View style={styles.metricDivider} />
          <View style={styles.metricItem}>
            <Text style={styles.metricLbl}>MAX HR</Text>
            <Text style={styles.metricVal}>{patientData.thalach} bpm</Text>
          </View>
        </View>

        {/* SHAP Feature Attribution Breakdown */}
        {Object.keys(shapValues).length > 0 && (
          <ShapBarChart shapValues={shapValues} patientData={patientData} />
        )}

        {/* Clinical Narrative Card */}
        {narrative ? (
          <View style={styles.narrativeCard}>
            <View style={styles.narrativeHeader}>
              <Ionicons name="document-text" size={16} color={COLORS.primaryLight} />
              <Text style={styles.narrativeTitle}>AI Diagnostic Narrative</Text>
            </View>
            <Text style={styles.narrativeText}>{narrative}</Text>
          </View>
        ) : null}

        {/* Action Button */}
        <TouchableOpacity style={styles.newAssessBtn} onPress={onNewAssessment} activeOpacity={0.8}>
          <Ionicons name="refresh" size={18} color={COLORS.white} />
          <Text style={styles.newAssessBtnText}>Perform Another Assessment</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 36,
  },
  emptyContainer: {
    flex: 1,
    backgroundColor: COLORS.background,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: COLORS.text,
    marginTop: 14,
  },
  emptySubtitle: {
    fontSize: 13,
    color: COLORS.textSecondary,
    marginTop: 6,
    textAlign: 'center',
  },
  startBtn: {
    backgroundColor: COLORS.primary,
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 12,
    marginTop: 20,
  },
  startBtnText: {
    fontSize: 13,
    fontWeight: '700',
    color: COLORS.white,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: COLORS.text,
  },
  headerSub: {
    fontSize: 12,
    color: COLORS.textSecondary,
    marginTop: 2,
  },
  saveBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: COLORS.surface,
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  saveBtnSaved: {
    borderColor: 'rgba(16, 185, 129, 0.4)',
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
  },
  saveBtnText: {
    fontSize: 12,
    fontWeight: '700',
    color: COLORS.text,
  },
  metricsStrip: {
    flexDirection: 'row',
    backgroundColor: COLORS.cardBg,
    borderRadius: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    marginTop: 4,
    marginBottom: 8,
  },
  metricItem: {
    flex: 1,
    alignItems: 'center',
  },
  metricDivider: {
    width: 1,
    backgroundColor: COLORS.surfaceBorder,
  },
  metricLbl: {
    fontSize: 9,
    fontWeight: '700',
    color: COLORS.textMuted,
    letterSpacing: 0.5,
  },
  metricVal: {
    fontSize: 12,
    fontWeight: '800',
    color: COLORS.text,
    marginTop: 2,
  },
  narrativeCard: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    marginTop: 12,
  },
  narrativeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  narrativeTitle: {
    fontSize: 13,
    fontWeight: '800',
    color: COLORS.text,
  },
  narrativeText: {
    fontSize: 12,
    color: COLORS.textSecondary,
    lineHeight: 18,
  },
  newAssessBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: COLORS.primary,
    paddingVertical: 14,
    borderRadius: 14,
    marginTop: 16,
  },
  newAssessBtnText: {
    fontSize: 14,
    fontWeight: '800',
    color: COLORS.white,
  },
});
