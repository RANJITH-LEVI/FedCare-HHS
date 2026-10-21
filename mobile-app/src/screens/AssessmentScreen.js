import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../constants/theme';
import apiService from '../services/apiService';
import ScanModal from '../components/ScanModal';

const INITIAL_FORM = {
  age: '55',
  sex: 1, // 0: Female, 1: Male
  cp: 3, // 1: Typical, 2: Atypical, 3: Non-Anginal, 4: Asymptomatic
  trestbps: '130',
  chol: '240',
  fbs: 0, // 0: <=120, 1: >120
  restecg: 0, // 0: Normal, 1: ST-T, 2: LVH
  thalach: '150',
  exang: 0, // 0: No, 1: Yes
  oldpeak: '1.0',
  slope: 1, // 1: Upsloping, 2: Flat, 3: Downsloping
  ca: '0',
  thal: 3, // 3: Normal, 6: Fixed Defect, 7: Reversible Defect
};

export default function AssessmentScreen({ onAssessmentComplete }) {
  const [form, setForm] = useState(INITIAL_FORM);
  const [scanModalVisible, setScanModalVisible] = useState(false);
  const [scannedBadge, setScannedBadge] = useState(false);
  const [evaluating, setEvaluating] = useState(false);

  const updateField = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleDataExtracted = (extracted) => {
    setForm((prev) => ({
      ...prev,
      age: extracted.age !== undefined ? String(extracted.age) : prev.age,
      sex: extracted.sex !== undefined ? Number(extracted.sex) : prev.sex,
      cp: extracted.cp !== undefined ? Number(extracted.cp) : prev.cp,
      trestbps: extracted.trestbps !== undefined ? String(extracted.trestbps) : prev.trestbps,
      chol: extracted.chol !== undefined ? String(extracted.chol) : prev.chol,
      fbs: extracted.fbs !== undefined ? Number(extracted.fbs) : prev.fbs,
      restecg: extracted.restecg !== undefined ? Number(extracted.restecg) : prev.restecg,
      thalach: extracted.thalach !== undefined ? String(extracted.thalach) : prev.thalach,
      exang: extracted.exang !== undefined ? Number(extracted.exang) : prev.exang,
      oldpeak: extracted.oldpeak !== undefined ? String(extracted.oldpeak) : prev.oldpeak,
      slope: extracted.slope !== undefined ? Number(extracted.slope) : prev.slope,
      ca: extracted.ca !== undefined ? String(extracted.ca) : prev.ca,
      thal: extracted.thal !== undefined ? Number(extracted.thal) : prev.thal,
    }));
    setScannedBadge(true);
  };

  const handleEvaluate = async () => {
    setEvaluating(true);
    try {
      const payload = {
        age: parseFloat(form.age) || 55,
        sex: parseInt(form.sex),
        cp: parseInt(form.cp),
        trestbps: parseFloat(form.trestbps) || 130,
        chol: parseFloat(form.chol) || 240,
        fbs: parseInt(form.fbs),
        restecg: parseInt(form.restecg),
        thalach: parseFloat(form.thalach) || 150,
        exang: parseInt(form.exang),
        oldpeak: parseFloat(form.oldpeak) || 1.0,
        slope: parseInt(form.slope),
        ca: parseInt(form.ca) || 0,
        thal: parseInt(form.thal),
      };

      // Call predict and explain in parallel
      const [predRes, explainRes] = await Promise.all([
        apiService.predict(payload),
        apiService.explain(payload).catch(() => ({})),
      ]);

      const assessmentResult = {
        patientData: payload,
        prediction: predRes.prediction,
        probability: predRes.probability,
        riskLevel: predRes.risk_level,
        confidence: predRes.confidence,
        shapValues: explainRes.shap_values || {},
        shapInteractions: explainRes.interactions || [],
        narrative: explainRes.narrative || '',
        scanned: scannedBadge,
      };

      setEvaluating(false);
      onAssessmentComplete(assessmentResult);
    } catch (err) {
      setEvaluating(false);
      Alert.alert('Evaluation Failed', err.message || 'Could not connect to model server.');
    }
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {/* Banner with Document Scan Trigger */}
        <View style={styles.scannerBanner}>
          <View style={styles.bannerLeft}>
            <View style={styles.bannerIconCircle}>
              <Ionicons name="scan" size={24} color={COLORS.primaryLight} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.bannerTitle}>Scan Medical Report</Text>
              <Text style={styles.bannerSubtitle}>Camera, gallery, or PDF auto-fill</Text>
            </View>
          </View>
          <TouchableOpacity
            style={styles.scanBtn}
            onPress={() => setScanModalVisible(true)}
            activeOpacity={0.8}
          >
            <Text style={styles.scanBtnText}>Scan Now</Text>
          </TouchableOpacity>
        </View>

        {scannedBadge && (
          <View style={styles.scannedBadgeWrap}>
            <Ionicons name="sparkles" size={14} color={COLORS.primaryLight} />
            <Text style={styles.scannedBadgeText}>
              Parameters auto-extracted via Gemini Vision. Verify below before evaluation.
            </Text>
          </View>
        )}

        {/* Section 1: Demographics */}
        <View style={styles.sectionCard}>
          <View style={styles.sectionHeader}>
            <Ionicons name="person" size={16} color={COLORS.primaryLight} />
            <Text style={styles.sectionTitle}>Patient Demographics</Text>
          </View>

          <View style={styles.formRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.fieldLabel}>Age (Years)</Text>
              <TextInput
                style={styles.input}
                value={form.age}
                onChangeText={(v) => updateField('age', v)}
                keyboardType="numeric"
                placeholder="55"
                placeholderTextColor={COLORS.textMuted}
              />
            </View>

            <View style={{ flex: 1 }}>
              <Text style={styles.fieldLabel}>Biological Sex</Text>
              <View style={styles.segmentedToggle}>
                <TouchableOpacity
                  style={[styles.segmentBtn, form.sex === 0 && styles.segmentBtnActive]}
                  onPress={() => updateField('sex', 0)}
                >
                  <Text style={[styles.segmentText, form.sex === 0 && styles.segmentTextActive]}>
                    Female
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.segmentBtn, form.sex === 1 && styles.segmentBtnActive]}
                  onPress={() => updateField('sex', 1)}
                >
                  <Text style={[styles.segmentText, form.sex === 1 && styles.segmentTextActive]}>
                    Male
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </View>

        {/* Section 2: Chest Symptoms */}
        <View style={styles.sectionCard}>
          <View style={styles.sectionHeader}>
            <Ionicons name="fitness" size={16} color={COLORS.primaryLight} />
            <Text style={styles.sectionTitle}>Chest Pain & Symptoms</Text>
          </View>

          <Text style={styles.fieldLabel}>Chest Pain Classification (CP)</Text>
          <View style={styles.radioGrid}>
            {[
              { id: 1, label: 'Typical Angina' },
              { id: 2, label: 'Atypical Angina' },
              { id: 3, label: 'Non-Anginal Pain' },
              { id: 4, label: 'Asymptomatic' },
            ].map((item) => (
              <TouchableOpacity
                key={item.id}
                style={[styles.radioItem, form.cp === item.id && styles.radioItemActive]}
                onPress={() => updateField('cp', item.id)}
                activeOpacity={0.7}
              >
                <View style={[styles.radioDot, form.cp === item.id && styles.radioDotActive]} />
                <Text style={[styles.radioLabel, form.cp === item.id && styles.radioLabelActive]}>
                  {item.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          <View style={{ marginTop: 12 }}>
            <Text style={styles.fieldLabel}>Exercise-Induced Angina</Text>
            <View style={styles.segmentedToggle}>
              <TouchableOpacity
                style={[styles.segmentBtn, form.exang === 0 && styles.segmentBtnActive]}
                onPress={() => updateField('exang', 0)}
              >
                <Text style={[styles.segmentText, form.exang === 0 && styles.segmentTextActive]}>
                  No (0)
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.segmentBtn, form.exang === 1 && styles.segmentBtnActive]}
                onPress={() => updateField('exang', 1)}
              >
                <Text style={[styles.segmentText, form.exang === 1 && styles.segmentTextActive]}>
                  Yes (1)
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>

        {/* Section 3: Vitals & Labs */}
        <View style={styles.sectionCard}>
          <View style={styles.sectionHeader}>
            <Ionicons name="pulse" size={16} color={COLORS.primaryLight} />
            <Text style={styles.sectionTitle}>Vitals & Blood Chemistry</Text>
          </View>

          <View style={styles.formRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.fieldLabel}>Resting BP (mm Hg)</Text>
              <TextInput
                style={styles.input}
                value={form.trestbps}
                onChangeText={(v) => updateField('trestbps', v)}
                keyboardType="numeric"
                placeholder="130"
                placeholderTextColor={COLORS.textMuted}
              />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.fieldLabel}>Serum Chol (mg/dl)</Text>
              <TextInput
                style={styles.input}
                value={form.chol}
                onChangeText={(v) => updateField('chol', v)}
                keyboardType="numeric"
                placeholder="240"
                placeholderTextColor={COLORS.textMuted}
              />
            </View>
          </View>

          <View style={styles.formRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.fieldLabel}>Max Heart Rate (bpm)</Text>
              <TextInput
                style={styles.input}
                value={form.thalach}
                onChangeText={(v) => updateField('thalach', v)}
                keyboardType="numeric"
                placeholder="150"
                placeholderTextColor={COLORS.textMuted}
              />
            </View>

            <View style={{ flex: 1 }}>
              <Text style={styles.fieldLabel}>Fasting Sugar &gt; 120</Text>
              <View style={styles.segmentedToggle}>
                <TouchableOpacity
                  style={[styles.segmentBtn, form.fbs === 0 && styles.segmentBtnActive]}
                  onPress={() => updateField('fbs', 0)}
                >
                  <Text style={[styles.segmentText, form.fbs === 0 && styles.segmentTextActive]}>
                    No (≤120)
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.segmentBtn, form.fbs === 1 && styles.segmentBtnActive]}
                  onPress={() => updateField('fbs', 1)}
                >
                  <Text style={[styles.segmentText, form.fbs === 1 && styles.segmentTextActive]}>
                    Yes (&gt;120)
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </View>

        {/* Section 4: ECG & Stress Biomarkers */}
        <View style={styles.sectionCard}>
          <View style={styles.sectionHeader}>
            <Ionicons name="analytics" size={16} color={COLORS.primaryLight} />
            <Text style={styles.sectionTitle}>Electrocardiogram & ST Dynamics</Text>
          </View>

          <Text style={styles.fieldLabel}>Resting ECG Result</Text>
          <View style={styles.segmentedToggle}>
            <TouchableOpacity
              style={[styles.segmentBtn, form.restecg === 0 && styles.segmentBtnActive]}
              onPress={() => updateField('restecg', 0)}
            >
              <Text style={[styles.segmentText, form.restecg === 0 && styles.segmentTextActive]}>
                Normal (0)
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.segmentBtn, form.restecg === 1 && styles.segmentBtnActive]}
              onPress={() => updateField('restecg', 1)}
            >
              <Text style={[styles.segmentText, form.restecg === 1 && styles.segmentTextActive]}>
                ST-T (1)
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.segmentBtn, form.restecg === 2 && styles.segmentBtnActive]}
              onPress={() => updateField('restecg', 2)}
            >
              <Text style={[styles.segmentText, form.restecg === 2 && styles.segmentTextActive]}>
                LVH (2)
              </Text>
            </TouchableOpacity>
          </View>

          <View style={[styles.formRow, { marginTop: 12 }]}>
            <View style={{ flex: 1 }}>
              <Text style={styles.fieldLabel}>ST Depression (oldpeak mm)</Text>
              <TextInput
                style={styles.input}
                value={form.oldpeak}
                onChangeText={(v) => updateField('oldpeak', v)}
                keyboardType="numeric"
                placeholder="1.0"
                placeholderTextColor={COLORS.textMuted}
              />
            </View>

            <View style={{ flex: 1 }}>
              <Text style={styles.fieldLabel}>ST Slope (Peak Exercise)</Text>
              <View style={styles.segmentedToggle}>
                <TouchableOpacity
                  style={[styles.segmentBtn, form.slope === 1 && styles.segmentBtnActive]}
                  onPress={() => updateField('slope', 1)}
                >
                  <Text style={[styles.segmentText, form.slope === 1 && styles.segmentTextActive]}>
                    Up
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.segmentBtn, form.slope === 2 && styles.segmentBtnActive]}
                  onPress={() => updateField('slope', 2)}
                >
                  <Text style={[styles.segmentText, form.slope === 2 && styles.segmentTextActive]}>
                    Flat
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.segmentBtn, form.slope === 3 && styles.segmentBtnActive]}
                  onPress={() => updateField('slope', 3)}
                >
                  <Text style={[styles.segmentText, form.slope === 3 && styles.segmentTextActive]}>
                    Down
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </View>

        {/* Section 5: Fluoroscopy & Thalassemia */}
        <View style={styles.sectionCard}>
          <View style={styles.sectionHeader}>
            <Ionicons name="medkit" size={16} color={COLORS.primaryLight} />
            <Text style={styles.sectionTitle}>Fluoroscopy & Thalassemia</Text>
          </View>

          <View style={styles.formRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.fieldLabel}>Major Vessels (0-3)</Text>
              <TextInput
                style={styles.input}
                value={form.ca}
                onChangeText={(v) => updateField('ca', v)}
                keyboardType="numeric"
                placeholder="0"
                placeholderTextColor={COLORS.textMuted}
              />
            </View>

            <View style={{ flex: 1 }}>
              <Text style={styles.fieldLabel}>Thalassemia Defect</Text>
              <View style={styles.segmentedToggle}>
                <TouchableOpacity
                  style={[styles.segmentBtn, form.thal === 3 && styles.segmentBtnActive]}
                  onPress={() => updateField('thal', 3)}
                >
                  <Text style={[styles.segmentText, form.thal === 3 && styles.segmentTextActive]}>
                    Normal (3)
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.segmentBtn, form.thal === 6 && styles.segmentBtnActive]}
                  onPress={() => updateField('thal', 6)}
                >
                  <Text style={[styles.segmentText, form.thal === 6 && styles.segmentTextActive]}>
                    Fixed (6)
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.segmentBtn, form.thal === 7 && styles.segmentBtnActive]}
                  onPress={() => updateField('thal', 7)}
                >
                  <Text style={[styles.segmentText, form.thal === 7 && styles.segmentTextActive]}>
                    Rev (7)
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </View>

        {/* Evaluate Action Button */}
        <TouchableOpacity
          style={styles.evaluateBtn}
          onPress={handleEvaluate}
          disabled={evaluating}
          activeOpacity={0.8}
        >
          {evaluating ? (
            <ActivityIndicator size="small" color={COLORS.white} />
          ) : (
            <>
              <Ionicons name="heart" size={20} color={COLORS.white} />
              <Text style={styles.evaluateBtnText}>Evaluate Cardiovascular Risk</Text>
            </>
          )}
        </TouchableOpacity>
      </ScrollView>

      {/* Document Scanner Modal */}
      <ScanModal
        visible={scanModalVisible}
        onClose={() => setScanModalVisible(false)}
        onDataExtracted={handleDataExtracted}
      />
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
  scannerBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: COLORS.surface,
    padding: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(56, 189, 248, 0.3)',
    marginBottom: 12,
  },
  bannerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1,
  },
  bannerIconCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(56, 189, 248, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  bannerTitle: {
    fontSize: 14,
    fontWeight: '800',
    color: COLORS.text,
  },
  bannerSubtitle: {
    fontSize: 11,
    color: COLORS.textSecondary,
    marginTop: 2,
  },
  scanBtn: {
    backgroundColor: COLORS.primary,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
  },
  scanBtnText: {
    fontSize: 12,
    fontWeight: '700',
    color: COLORS.white,
  },
  scannedBadgeWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: 'rgba(56, 189, 248, 0.12)',
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(56, 189, 248, 0.25)',
    marginBottom: 12,
  },
  scannedBadgeText: {
    fontSize: 11,
    color: COLORS.primaryLight,
    fontWeight: '600',
    flex: 1,
  },
  sectionCard: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    marginBottom: 12,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: '800',
    color: COLORS.text,
    letterSpacing: 0.3,
  },
  formRow: {
    flexDirection: 'row',
    gap: 12,
  },
  fieldLabel: {
    fontSize: 11,
    fontWeight: '600',
    color: COLORS.textSecondary,
    marginBottom: 6,
  },
  input: {
    backgroundColor: COLORS.inputBg,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    paddingHorizontal: 12,
    paddingVertical: 9,
    fontSize: 13,
    color: COLORS.text,
  },
  segmentedToggle: {
    flexDirection: 'row',
    backgroundColor: COLORS.inputBg,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    padding: 2,
  },
  segmentBtn: {
    flex: 1,
    paddingVertical: 7,
    alignItems: 'center',
    borderRadius: 8,
  },
  segmentBtnActive: {
    backgroundColor: COLORS.primary,
  },
  segmentText: {
    fontSize: 11,
    fontWeight: '600',
    color: COLORS.textSecondary,
  },
  segmentTextActive: {
    color: COLORS.white,
    fontWeight: '700',
  },
  radioGrid: {
    gap: 8,
  },
  radioItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: COLORS.inputBg,
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  radioItemActive: {
    borderColor: COLORS.primaryLight,
    backgroundColor: 'rgba(56, 189, 248, 0.08)',
  },
  radioDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: COLORS.textMuted,
  },
  radioDotActive: {
    borderColor: COLORS.primaryLight,
    backgroundColor: COLORS.primaryLight,
  },
  radioLabel: {
    fontSize: 12,
    color: COLORS.textSecondary,
  },
  radioLabelActive: {
    color: COLORS.text,
    fontWeight: '700',
  },
  evaluateBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    backgroundColor: COLORS.danger,
    paddingVertical: 14,
    borderRadius: 14,
    marginTop: 6,
    shadowColor: COLORS.danger,
    shadowOpacity: 0.35,
    shadowRadius: 10,
    elevation: 4,
  },
  evaluateBtnText: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.white,
    letterSpacing: 0.3,
  },
});
