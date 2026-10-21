import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  TextInput,
  RefreshControl,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, RISK_LEVELS } from '../constants/theme';
import storageService from '../services/storageService';
import apiService from '../services/apiService';

export default function HistoryScreen({ onSelectRecord, onNewAssessment }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const loadHistory = async () => {
    try {
      const localData = await storageService.getAssessments();
      const remoteData = await apiService.getPredictionHistory().catch(() => []);

      // Format remote backend predictions to match local history format
      const formattedRemote = remoteData.map((p) => {
        const patientData = p.patient_data || {};
        const probability = p.risk_score ?? 0;
        let riskLevel = p.risk_tier || 'Low Risk';

        return {
          id: `remote_${p.prediction_id}`,
          timestamp: p.created_at,
          patientData,
          probability,
          riskLevel,
          confidence: 0.88,
          shapValues: p.shap_values || {},
          narrative: p.plain_language_narrative || '',
          scanned: false,
          isRemote: true,
        };
      });

      // Merge local and remote history
      const combined = [...localData, ...formattedRemote];
      setHistory(combined);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleDelete = (id) => {
    Alert.alert('Delete Assessment', 'Are you sure you want to remove this patient assessment?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          if (!id.startsWith('remote_')) {
            const updated = await storageService.deleteAssessment(id);
            setHistory(updated);
          } else {
            setHistory(prev => prev.filter(item => item.id !== id));
          }
        },
      },
    ]);
  };

  const handleClearAll = () => {
    if (history.length === 0) return;
    Alert.alert('Clear History', 'Delete all locally saved patient evaluations?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Clear All',
        style: 'destructive',
        onPress: async () => {
          await storageService.clearAll();
          setHistory([]);
        },
      },
    ]);
  };

  const filteredHistory = history.filter((item) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    const p = item.patientData || {};
    return (
      item.riskLevel?.toLowerCase().includes(q) ||
      String(p.age).includes(q) ||
      String(p.trestbps).includes(q) ||
      String(p.chol).includes(q)
    );
  });

  return (
    <View style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={() => {
              setRefreshing(true);
              loadHistory();
            }}
            tintColor={COLORS.primaryLight}
          />
        }
      >
        <View style={styles.headerRow}>
          <View>
            <Text style={styles.headerTitle}>Patient History</Text>
            <Text style={styles.headerSub}>{history.length} Saved Clinical Evaluations</Text>
          </View>

          {history.length > 0 && (
            <TouchableOpacity style={styles.clearBtn} onPress={handleClearAll} activeOpacity={0.7}>
              <Ionicons name="trash-outline" size={16} color={COLORS.danger} />
              <Text style={styles.clearBtnText}>Clear</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Search Bar */}
        {history.length > 0 && (
          <View style={styles.searchBar}>
            <Ionicons name="search" size={16} color={COLORS.textMuted} />
            <TextInput
              style={styles.searchInput}
              placeholder="Search by risk level, age, BP..."
              placeholderTextColor={COLORS.textMuted}
              value={searchQuery}
              onChangeText={setSearchQuery}
            />
          </View>
        )}

        {filteredHistory.length === 0 ? (
          <View style={styles.emptyWrap}>
            <Ionicons name="folder-open-outline" size={48} color={COLORS.textMuted} />
            <Text style={styles.emptyTitle}>
              {history.length === 0 ? 'No Assessments Saved Yet' : 'No Matching Records'}
            </Text>
            <Text style={styles.emptyDesc}>
              {history.length === 0
                ? 'Assessments saved from the evaluation screen will be stored here on your device and synchronized with the backend.'
                : 'Try adjusting your search criteria.'}
            </Text>
            {history.length === 0 && (
              <TouchableOpacity style={styles.startBtn} onPress={onNewAssessment}>
                <Text style={styles.startBtnText}>Start First Assessment</Text>
              </TouchableOpacity>
            )}
          </View>
        ) : (
          <View style={styles.recordsList}>
            {filteredHistory.map((item) => {
              const p = item.patientData || {};
              const levelInfo = RISK_LEVELS[item.riskLevel] || RISK_LEVELS['Low Risk'];
              const dateStr = item.timestamp
                ? new Date(item.timestamp).toLocaleDateString(undefined, {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                  })
                : 'Recent';

              return (
                <TouchableOpacity
                  key={item.id}
                  style={[styles.recordCard, { borderColor: levelInfo.border }]}
                  onPress={() => onSelectRecord(item)}
                  activeOpacity={0.8}
                >
                  <View style={styles.cardHeader}>
                    <View style={styles.cardHeaderLeft}>
                      <View style={[styles.badge, { backgroundColor: levelInfo.bg }]}>
                        <Ionicons name={levelInfo.icon} size={14} color={levelInfo.color} />
                        <Text style={[styles.badgeText, { color: levelInfo.color }]}>
                          {item.riskLevel}
                        </Text>
                      </View>
                      {item.scanned && (
                        <View style={styles.scannedPill}>
                          <Ionicons name="scan" size={10} color={COLORS.primaryLight} />
                          <Text style={styles.scannedPillText}>Vision Scan</Text>
                        </View>
                      )}
                      {item.isRemote && (
                        <View style={[styles.scannedPill, { backgroundColor: 'rgba(99, 102, 241, 0.15)' }]}>
                          <Ionicons name="cloud" size={10} color={COLORS.accent} />
                          <Text style={[styles.scannedPillText, { color: COLORS.accent }]}>Cloud Sync</Text>
                        </View>
                      )}
                    </View>

                    <TouchableOpacity onPress={() => handleDelete(item.id)}>
                      <Ionicons name="close" size={18} color={COLORS.textMuted} />
                    </TouchableOpacity>
                  </View>

                  <View style={styles.cardBody}>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.patientInfo}>
                        Patient • Age {p.age || 55}, {p.sex === 1 ? 'Male' : 'Female'}
                      </Text>
                      <Text style={styles.patientVitals}>
                        BP: {p.trestbps || 130} mmHg • Chol: {p.chol || 240} mg/dl • MaxHR: {p.thalach || 150} bpm
                      </Text>
                      <Text style={styles.dateText}>{dateStr}</Text>
                    </View>

                    <View style={styles.probBox}>
                      <Text style={[styles.probVal, { color: levelInfo.color }]}>
                        {Math.round((item.probability || 0) * 100)}%
                      </Text>
                      <Text style={styles.probLbl}>Probability</Text>
                    </View>
                  </View>
                </TouchableOpacity>
              );
            })}
          </View>
        )}
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
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
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
  clearBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
  },
  clearBtnText: {
    fontSize: 11,
    fontWeight: '700',
    color: COLORS.danger,
  },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: COLORS.cardBg,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    paddingHorizontal: 12,
    height: 40,
    marginBottom: 16,
  },
  searchInput: {
    flex: 1,
    fontSize: 12,
    color: COLORS.text,
  },
  emptyWrap: {
    alignItems: 'center',
    paddingVertical: 50,
  },
  emptyTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: COLORS.text,
    marginTop: 14,
  },
  emptyDesc: {
    fontSize: 12,
    color: COLORS.textSecondary,
    marginTop: 6,
    textAlign: 'center',
    paddingHorizontal: 30,
    lineHeight: 18,
  },
  startBtn: {
    backgroundColor: COLORS.primary,
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 10,
    marginTop: 20,
  },
  startBtnText: {
    fontSize: 12,
    fontWeight: '700',
    color: COLORS.white,
  },
  recordsList: {
    gap: 12,
  },
  recordCard: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  cardHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },
  badgeText: {
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  scannedPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(56, 189, 248, 0.15)',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 6,
  },
  scannedPillText: {
    fontSize: 9,
    fontWeight: '700',
    color: COLORS.primaryLight,
  },
  cardBody: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  patientInfo: {
    fontSize: 13,
    fontWeight: '700',
    color: COLORS.text,
  },
  patientVitals: {
    fontSize: 11,
    color: COLORS.textSecondary,
    marginTop: 3,
  },
  dateText: {
    fontSize: 10,
    color: COLORS.textMuted,
    marginTop: 4,
  },
  probBox: {
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 10,
  },
  probVal: {
    fontSize: 18,
    fontWeight: '900',
  },
  probLbl: {
    fontSize: 9,
    color: COLORS.textMuted,
    marginTop: 1,
  },
});
