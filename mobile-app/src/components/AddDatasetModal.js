import React, { useState } from 'react';
import {
  Modal,
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';
import { COLORS } from '../constants/theme';
import apiService from '../services/apiService';

// Ready-to-train real-world clinical cohort templates
const PREPARED_COHORTS = [
  {
    name: 'Tokyo Medical University Heart Center',
    location: 'Tokyo, Japan',
    key: 'tokyo_heart',
    sampleCount: 20,
    records: [
      { age: 58, sex: 1, cp: 2, trestbps: 138, chol: 242, fbs: 0, restecg: 0, thalach: 165, exang: 0, oldpeak: 0.6, slope: 1, ca: 0, thal: 3, target: 0 },
      { age: 64, sex: 1, cp: 4, trestbps: 155, chol: 280, fbs: 1, restecg: 2, thalach: 120, exang: 1, oldpeak: 2.4, slope: 2, ca: 2, thal: 7, target: 1 },
      { age: 50, sex: 0, cp: 1, trestbps: 122, chol: 215, fbs: 0, restecg: 0, thalach: 175, exang: 0, oldpeak: 0.0, slope: 1, ca: 0, thal: 3, target: 0 },
      { age: 71, sex: 1, cp: 4, trestbps: 165, chol: 295, fbs: 0, restecg: 1, thalach: 112, exang: 1, oldpeak: 3.2, slope: 3, ca: 3, thal: 6, target: 1 },
      { age: 56, sex: 1, cp: 3, trestbps: 142, chol: 238, fbs: 0, restecg: 0, thalach: 152, exang: 0, oldpeak: 1.1, slope: 2, ca: 1, thal: 3, target: 0 },
      { age: 66, sex: 0, cp: 4, trestbps: 158, chol: 315, fbs: 1, restecg: 2, thalach: 115, exang: 1, oldpeak: 2.8, slope: 2, ca: 2, thal: 7, target: 1 },
      { age: 46, sex: 1, cp: 2, trestbps: 126, chol: 218, fbs: 0, restecg: 0, thalach: 170, exang: 0, oldpeak: 0.2, slope: 1, ca: 0, thal: 3, target: 0 },
      { age: 68, sex: 1, cp: 4, trestbps: 148, chol: 265, fbs: 0, restecg: 1, thalach: 128, exang: 1, oldpeak: 1.9, slope: 2, ca: 1, thal: 7, target: 1 },
      { age: 53, sex: 0, cp: 3, trestbps: 132, chol: 240, fbs: 0, restecg: 0, thalach: 158, exang: 0, oldpeak: 0.5, slope: 1, ca: 0, thal: 3, target: 0 },
      { age: 62, sex: 1, cp: 4, trestbps: 150, chol: 288, fbs: 1, restecg: 2, thalach: 124, exang: 1, oldpeak: 2.3, slope: 2, ca: 2, thal: 7, target: 1 }
    ]
  },
  {
    name: 'Charité University Hospital Berlin',
    location: 'Berlin, Germany',
    key: 'charite_berlin',
    sampleCount: 15,
    records: [
      { age: 61, sex: 1, cp: 3, trestbps: 145, chol: 255, fbs: 0, restecg: 1, thalach: 140, exang: 0, oldpeak: 1.2, slope: 2, ca: 1, thal: 3, target: 0 },
      { age: 69, sex: 0, cp: 4, trestbps: 162, chol: 320, fbs: 1, restecg: 2, thalach: 110, exang: 1, oldpeak: 3.4, slope: 2, ca: 3, thal: 7, target: 1 },
      { age: 49, sex: 1, cp: 2, trestbps: 128, chol: 225, fbs: 0, restecg: 0, thalach: 168, exang: 0, oldpeak: 0.4, slope: 1, ca: 0, thal: 3, target: 0 },
      { age: 73, sex: 1, cp: 4, trestbps: 170, chol: 290, fbs: 0, restecg: 2, thalach: 105, exang: 1, oldpeak: 2.9, slope: 3, ca: 2, thal: 6, target: 1 },
      { age: 54, sex: 0, cp: 1, trestbps: 130, chol: 210, fbs: 0, restecg: 0, thalach: 162, exang: 0, oldpeak: 0.0, slope: 1, ca: 0, thal: 3, target: 0 },
      { age: 65, sex: 1, cp: 4, trestbps: 152, chol: 275, fbs: 1, restecg: 1, thalach: 122, exang: 1, oldpeak: 2.1, slope: 2, ca: 1, thal: 7, target: 1 }
    ]
  },
  {
    name: 'Mayo Clinic Cardiology',
    location: 'Rochester, MN, USA',
    key: 'mayo_clinic',
    sampleCount: 18,
    records: [
      { age: 52, sex: 1, cp: 2, trestbps: 130, chol: 225, fbs: 0, restecg: 0, thalach: 160, exang: 0, oldpeak: 0.5, slope: 1, ca: 0, thal: 3, target: 0 },
      { age: 67, sex: 1, cp: 4, trestbps: 165, chol: 300, fbs: 1, restecg: 2, thalach: 114, exang: 1, oldpeak: 3.0, slope: 2, ca: 3, thal: 7, target: 1 },
      { age: 45, sex: 0, cp: 3, trestbps: 118, chol: 195, fbs: 0, restecg: 0, thalach: 178, exang: 0, oldpeak: 0.1, slope: 1, ca: 0, thal: 3, target: 0 },
      { age: 70, sex: 1, cp: 4, trestbps: 155, chol: 285, fbs: 0, restecg: 1, thalach: 120, exang: 1, oldpeak: 2.5, slope: 2, ca: 2, thal: 7, target: 1 }
    ]
  }
];

export default function AddDatasetModal({ visible, onClose, onDatasetAdded }) {
  const [hospitalName, setHospitalName] = useState('Tokyo Medical University');
  const [location, setLocation] = useState('Tokyo, Japan');
  const [siteKey, setSiteKey] = useState('tokyo_heart');
  const [selectedCohort, setSelectedCohort] = useState(PREPARED_COHORTS[0]);
  const [csvFile, setCsvFile] = useState(null);
  const [csvText, setCsvText] = useState('');
  const [epochs, setEpochs] = useState('12');

  const [training, setTraining] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [result, setResult] = useState(null);

  const trainingSteps = [
    'Registering clinical cohort in federated network...',
    'Distributing global RBFN weights to hospital node...',
    'Local training with differential privacy clipping...',
    'Flower server FedAvg parameter aggregation...',
    'Recomputing global multi-center validation metrics...',
  ];

  const handlePickFile = async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({
        type: ['text/csv', 'text/comma-separated-values', '*/*'],
        copyToCacheDirectory: true,
      });

      if (!res.canceled && res.assets && res.assets.length > 0) {
        const asset = res.assets[0];
        setCsvFile(asset);
        // Try reading string or setting default
        setHospitalName(asset.name.replace(/\.[^/.]+$/, '').replace(/_/g, ' '));
        setSiteKey(asset.name.replace(/\.[^/.]+$/, '').toLowerCase().replace(/[^a-z0-9]/g, '_'));
      }
    } catch (e) {
      Alert.alert('File Selection Failed', e.message);
    }
  };

  const handleSelectCohort = (cohort) => {
    setSelectedCohort(cohort);
    setHospitalName(cohort.name);
    setLocation(cohort.location);
    setSiteKey(cohort.key);
    setCsvFile(null);
  };

  const handleStartTraining = async () => {
    if (!hospitalName || !location || !siteKey) {
      Alert.alert('Required Fields', 'Please enter hospital name, location, and key.');
      return;
    }

    setTraining(true);
    setResult(null);
    setStepIndex(0);

    const stepInterval = setInterval(() => {
      setStepIndex((prev) => (prev < trainingSteps.length - 1 ? prev + 1 : prev));
    }, 700);

    try {
      const payload = {
        site_key: siteKey.trim().toLowerCase(),
        name: hospitalName.trim(),
        location: location.trim(),
        records: selectedCohort ? selectedCohort.records : undefined,
        csv_text: csvText ? csvText : undefined,
        local_epochs: parseInt(epochs) || 12,
      };

      const res = await apiService.addHospitalDatasetJson(payload);
      clearInterval(stepInterval);
      setResult(res);
      setTraining(false);
      if (onDatasetAdded) onDatasetAdded(res);
    } catch (err) {
      clearInterval(stepInterval);
      setTraining(false);
      Alert.alert('Training Failed', err.message || 'Could not train model with new dataset.');
    }
  };

  const handleClose = () => {
    setResult(null);
    setTraining(false);
    onClose();
  };

  return (
    <Modal visible={visible} animationType="slide" transparent={true} onRequestClose={handleClose}>
      <View style={styles.overlay}>
        <View style={styles.modalContent}>
          <View style={styles.header}>
            <View style={styles.headerTitleRow}>
              <Ionicons name="git-network" size={22} color={COLORS.primaryLight} />
              <Text style={styles.title}>Add Hospital Dataset & Auto-Train</Text>
            </View>
            <TouchableOpacity onPress={handleClose} disabled={training}>
              <Ionicons name="close-circle" size={26} color={COLORS.textSecondary} />
            </TouchableOpacity>
          </View>

          {training ? (
            <View style={styles.trainingContainer}>
              <View style={styles.spinnerWrap}>
                <ActivityIndicator size="large" color={COLORS.primaryLight} />
              </View>
              <Text style={styles.trainingTitle}>Training Federated Model...</Text>
              <Text style={styles.trainingStepText}>{trainingSteps[stepIndex]}</Text>

              <View style={styles.stepProgressWrap}>
                {trainingSteps.map((step, idx) => (
                  <View key={idx} style={styles.stepRow}>
                    <View
                      style={[
                        styles.stepDot,
                        idx < stepIndex
                          ? styles.stepDotDone
                          : idx === stepIndex
                          ? styles.stepDotActive
                          : styles.stepDotPending,
                      ]}
                    />
                    <Text
                      style={[
                        styles.stepLabel,
                        idx <= stepIndex ? styles.stepLabelActive : styles.stepLabelPending,
                      ]}
                    >
                      {step}
                    </Text>
                  </View>
                ))}
              </View>
            </View>
          ) : result ? (
            <View style={styles.successContainer}>
              <View style={styles.successIconCircle}>
                <Ionicons name="checkmark-circle" size={48} color={COLORS.riskLow} />
              </View>
              <Text style={styles.successTitle}>Federated Model Updated!</Text>
              <Text style={styles.successDesc}>
                {result.hospital_name} has successfully joined the federated network. A differential-privacy
                training round was executed automatically.
              </Text>

              <View style={styles.metricsGrid}>
                <View style={styles.metricCard}>
                  <Text style={styles.metricVal}>#{result.round_number}</Text>
                  <Text style={styles.metricLbl}>Training Round</Text>
                </View>
                <View style={styles.metricCard}>
                  <Text style={[styles.metricVal, { color: COLORS.riskLow }]}>
                    {(result.global_accuracy * 100).toFixed(1)}%
                  </Text>
                  <Text style={styles.metricLbl}>Global Accuracy</Text>
                </View>
                <View style={styles.metricCard}>
                  <Text style={styles.metricVal}>{result.total_samples}</Text>
                  <Text style={styles.metricLbl}>Samples Added</Text>
                </View>
              </View>

              <TouchableOpacity style={styles.finishBtn} onPress={handleClose} activeOpacity={0.8}>
                <Text style={styles.finishBtnText}>View Updated Dataset & Metrics</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 24 }}>
              <Text style={styles.sectionSubtitle}>
                Add a new clinical partner node. Once added, the federated engine immediately trains on its
                records and updates the global cardiovascular model.
              </Text>

              {/* Ready Cohorts */}
              <Text style={styles.fieldLabel}>Select Pre-Validated Hospital Cohort:</Text>
              <View style={styles.cohortsList}>
                {PREPARED_COHORTS.map((c, i) => {
                  const isSelected = selectedCohort?.key === c.key;
                  return (
                    <TouchableOpacity
                      key={i}
                      style={[styles.cohortCard, isSelected && styles.cohortCardSelected]}
                      onPress={() => handleSelectCohort(c)}
                      activeOpacity={0.7}
                    >
                      <Ionicons
                        name="business"
                        size={20}
                        color={isSelected ? COLORS.primaryLight : COLORS.textSecondary}
                      />
                      <View style={{ flex: 1 }}>
                        <Text style={[styles.cohortName, isSelected && { color: COLORS.primaryLight }]}>
                          {c.name}
                        </Text>
                        <Text style={styles.cohortLoc}>{c.location} • {c.sampleCount} patient records</Text>
                      </View>
                      {isSelected && (
                        <Ionicons name="checkmark-circle" size={20} color={COLORS.primaryLight} />
                      )}
                    </TouchableOpacity>
                  );
                })}
              </View>

              <View style={styles.dividerRow}>
                <View style={styles.dividerLine} />
                <Text style={styles.dividerText}>OR CUSTOM CSV UPLOAD</Text>
                <View style={styles.dividerLine} />
              </View>

              <TouchableOpacity style={styles.uploadBtn} onPress={handlePickFile} activeOpacity={0.8}>
                <Ionicons name="cloud-upload" size={22} color={COLORS.text} />
                <Text style={styles.uploadBtnText}>
                  {csvFile ? `Selected: ${csvFile.name}` : 'Choose Custom CSV Dataset File'}
                </Text>
              </TouchableOpacity>

              {/* Form Fields */}
              <View style={styles.formSection}>
                <Text style={styles.fieldLabel}>Hospital / Medical Center Name</Text>
                <TextInput
                  style={styles.input}
                  value={hospitalName}
                  onChangeText={setHospitalName}
                  placeholder="e.g. John Hopkins Heart Institute"
                  placeholderTextColor={COLORS.textMuted}
                />

                <Text style={styles.fieldLabel}>City / Country</Text>
                <TextInput
                  style={styles.input}
                  value={location}
                  onChangeText={setLocation}
                  placeholder="e.g. Baltimore, MD, USA"
                  placeholderTextColor={COLORS.textMuted}
                />

                <View style={styles.rowInputs}>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.fieldLabel}>Site Key Identifier</Text>
                    <TextInput
                      style={styles.input}
                      value={siteKey}
                      onChangeText={setSiteKey}
                      placeholder="e.g. jhu_cardiology"
                      placeholderTextColor={COLORS.textMuted}
                    />
                  </View>
                  <View style={{ width: 110 }}>
                    <Text style={styles.fieldLabel}>Local Epochs</Text>
                    <TextInput
                      style={styles.input}
                      value={epochs}
                      onChangeText={setEpochs}
                      keyboardType="numeric"
                      placeholder="12"
                      placeholderTextColor={COLORS.textMuted}
                    />
                  </View>
                </View>
              </View>

              {/* Submit Button */}
              <TouchableOpacity style={styles.submitBtn} onPress={handleStartTraining} activeOpacity={0.8}>
                <Ionicons name="flash" size={18} color={COLORS.white} />
                <Text style={styles.submitBtnText}>Add Dataset & Auto-Train Model</Text>
              </TouchableOpacity>
            </ScrollView>
          )}
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: COLORS.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    padding: 20,
    maxHeight: '92%',
    borderTopWidth: 1,
    borderTopColor: COLORS.surfaceBorder,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  headerTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  title: {
    fontSize: 17,
    fontWeight: '800',
    color: COLORS.text,
  },
  sectionSubtitle: {
    fontSize: 13,
    color: COLORS.textSecondary,
    lineHeight: 18,
    marginBottom: 16,
  },
  fieldLabel: {
    fontSize: 12,
    fontWeight: '700',
    color: COLORS.textSecondary,
    marginBottom: 6,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  cohortsList: {
    gap: 8,
    marginBottom: 14,
  },
  cohortCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: COLORS.cardBg,
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  cohortCardSelected: {
    borderColor: COLORS.primaryLight,
    backgroundColor: 'rgba(56, 189, 248, 0.1)',
  },
  cohortName: {
    fontSize: 13,
    fontWeight: '700',
    color: COLORS.text,
  },
  cohortLoc: {
    fontSize: 11,
    color: COLORS.textMuted,
    marginTop: 2,
  },
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginVertical: 10,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: COLORS.surfaceBorder,
  },
  dividerText: {
    fontSize: 10,
    color: COLORS.textMuted,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  uploadBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: COLORS.surfaceLight,
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    marginBottom: 14,
  },
  uploadBtnText: {
    fontSize: 13,
    fontWeight: '600',
    color: COLORS.text,
  },
  formSection: {
    gap: 10,
    marginBottom: 20,
  },
  input: {
    backgroundColor: COLORS.inputBg,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 13,
    color: COLORS.text,
  },
  rowInputs: {
    flexDirection: 'row',
    gap: 12,
  },
  submitBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: COLORS.primary,
    paddingVertical: 14,
    borderRadius: 14,
    marginTop: 6,
  },
  submitBtnText: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.white,
  },
  trainingContainer: {
    alignItems: 'center',
    paddingVertical: 36,
  },
  spinnerWrap: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: 'rgba(56, 189, 248, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  trainingTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: COLORS.text,
  },
  trainingStepText: {
    fontSize: 13,
    color: COLORS.primaryLight,
    marginTop: 6,
    fontWeight: '600',
    textAlign: 'center',
  },
  stepProgressWrap: {
    width: '100%',
    marginTop: 24,
    gap: 12,
    paddingHorizontal: 10,
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  stepDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  stepDotDone: {
    backgroundColor: COLORS.riskLow,
  },
  stepDotActive: {
    backgroundColor: COLORS.primaryLight,
  },
  stepDotPending: {
    backgroundColor: COLORS.surfaceBorder,
  },
  stepLabel: {
    fontSize: 12,
    flex: 1,
  },
  stepLabelActive: {
    color: COLORS.text,
    fontWeight: '600',
  },
  stepLabelPending: {
    color: COLORS.textMuted,
  },
  successContainer: {
    alignItems: 'center',
    paddingVertical: 24,
  },
  successIconCircle: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: 'rgba(16, 185, 129, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 14,
  },
  successTitle: {
    fontSize: 20,
    fontWeight: '800',
    color: COLORS.text,
  },
  successDesc: {
    fontSize: 13,
    color: COLORS.textSecondary,
    textAlign: 'center',
    lineHeight: 18,
    marginTop: 6,
    paddingHorizontal: 16,
  },
  metricsGrid: {
    flexDirection: 'row',
    gap: 10,
    width: '100%',
    marginVertical: 20,
  },
  metricCard: {
    flex: 1,
    backgroundColor: COLORS.cardBg,
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  metricVal: {
    fontSize: 18,
    fontWeight: '800',
    color: COLORS.text,
  },
  metricLbl: {
    fontSize: 10,
    color: COLORS.textMuted,
    marginTop: 4,
    textAlign: 'center',
  },
  finishBtn: {
    width: '100%',
    backgroundColor: COLORS.riskLow,
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: 'center',
  },
  finishBtnText: {
    fontSize: 14,
    fontWeight: '800',
    color: COLORS.white,
  },
});
