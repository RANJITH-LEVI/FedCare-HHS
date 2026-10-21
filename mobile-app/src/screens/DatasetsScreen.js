import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
  TextInput,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../constants/theme';
import apiService from '../services/apiService';
import AddDatasetModal from '../components/AddDatasetModal';

export default function DatasetsScreen() {
  const [datasets, setDatasets] = useState([]);
  const [selectedSiteKey, setSelectedSiteKey] = useState('cleveland');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filterTarget, setFilterTarget] = useState('all'); // 'all', 'positive', 'negative'
  const [searchQuery, setSearchQuery] = useState('');
  const [addModalVisible, setAddModalVisible] = useState(false);
  const [metrics, setMetrics] = useState(null);

  const fetchDatasets = async () => {
    try {
      const [data, met] = await Promise.all([
        apiService.getHospitalDatasets(),
        apiService.getModelMetrics().catch(() => null),
      ]);
      setDatasets(data || []);
      setMetrics(met);
      if (data && data.length > 0 && !selectedSiteKey) {
        setSelectedSiteKey(data[0].site_key);
      }
    } catch (e) {
      console.error('Failed to load hospital datasets', e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchDatasets();
  };

  const handleDatasetAdded = (res) => {
    fetchDatasets();
    if (res && res.site_key) {
      setSelectedSiteKey(res.site_key);
    }
  };

  const currentDataset = datasets.find((d) => d.site_key === selectedSiteKey) || datasets[0];

  // Filter records in current dataset
  const filteredRecords = (currentDataset?.records || []).filter((rec) => {
    if (filterTarget === 'positive' && rec.target <= 0) return false;
    if (filterTarget === 'negative' && rec.target > 0) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      const matchAge = String(rec.age).includes(q);
      const matchBp = String(rec.trestbps).includes(q);
      const matchChol = String(rec.chol).includes(q);
      return matchAge || matchBp || matchChol;
    }
    return true;
  });

  return (
    <View style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} tintColor={COLORS.primaryLight} />
        }
      >
        {/* Screen Header */}
        <View style={styles.headerRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.headerTitle}>Hospital Datasets & FL</Text>
            <Text style={styles.headerSub}>Decentralized Multi-Center Cohorts</Text>
          </View>

          <TouchableOpacity
            style={styles.addBtn}
            onPress={() => setAddModalVisible(true)}
            activeOpacity={0.8}
          >
            <Ionicons name="add-circle" size={18} color={COLORS.white} />
            <Text style={styles.addBtnText}>Add Dataset</Text>
          </TouchableOpacity>
        </View>

        {/* Global Model Telemetry Banner */}
        {metrics && (
          <View style={styles.telemetryBanner}>
            <View style={styles.telemetryLeft}>
              <Ionicons name="hardware-chip" size={20} color={COLORS.primaryLight} />
              <View>
                <Text style={styles.telemetryTitle}>
                  FL Global Model • Round #{metrics.latest_round}
                </Text>
                <Text style={styles.telemetrySub}>
                  Differential Privacy ε = {metrics.differential_privacy?.epsilon?.toFixed(2) || '0.84'}
                </Text>
              </View>
            </View>
            <View style={styles.telemetryRight}>
              <Text style={styles.telemetryAcc}>
                {((metrics.global_metrics?.accuracy || 0.78) * 100).toFixed(1)}%
              </Text>
              <Text style={styles.telemetryAccLabel}>Global Acc</Text>
            </View>
          </View>
        )}

        {loading ? (
          <View style={styles.loadingWrap}>
            <ActivityIndicator size="large" color={COLORS.primaryLight} />
            <Text style={styles.loadingText}>Loading hospital cohorts...</Text>
          </View>
        ) : (
          <>
            {/* Hospital Selector Horizontal Scroll */}
            <Text style={styles.sectionLabel}>Select Hospital Partner Node</Text>
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.hospitalTabs}
            >
              {datasets.map((item) => {
                const isSelected = item.site_key === selectedSiteKey;
                return (
                  <TouchableOpacity
                    key={item.site_key}
                    style={[styles.hospTab, isSelected && styles.hospTabSelected]}
                    onPress={() => setSelectedSiteKey(item.site_key)}
                    activeOpacity={0.7}
                  >
                    <Ionicons
                      name="business"
                      size={16}
                      color={isSelected ? COLORS.white : COLORS.textSecondary}
                    />
                    <View>
                      <Text style={[styles.hospTabName, isSelected && styles.hospTabNameSelected]}>
                        {item.name.replace(' Foundation', '').replace(' Cardiology', '')}
                      </Text>
                      <Text style={styles.hospTabCount}>{item.total_records} patients</Text>
                    </View>
                  </TouchableOpacity>
                );
              })}
            </ScrollView>

            {/* Current Hospital Cohort Card */}
            {currentDataset && (
              <View style={styles.cohortCard}>
                <View style={styles.cohortTop}>
                  <View style={{ flex: 1 }}>
                    <View style={styles.cohortBadgeRow}>
                      <View style={styles.hospIdBadge}>
                        <Text style={styles.hospIdText}>{currentDataset.id}</Text>
                      </View>
                      <Text style={styles.cohortLoc}>{currentDataset.location}</Text>
                    </View>
                    <Text style={styles.cohortTitle}>{currentDataset.name}</Text>
                  </View>
                </View>

                {/* Cohort Stats Grid */}
                <View style={styles.statsGrid}>
                  <View style={styles.statBox}>
                    <Text style={styles.statVal}>{currentDataset.total_records}</Text>
                    <Text style={styles.statLbl}>Total Patients</Text>
                  </View>
                  <View style={styles.statBox}>
                    <Text style={[styles.statVal, { color: COLORS.riskHigh }]}>
                      {currentDataset.positive_records}
                    </Text>
                    <Text style={styles.statLbl}>Disease Present</Text>
                  </View>
                  <View style={styles.statBox}>
                    <Text style={[styles.statVal, { color: COLORS.riskLow }]}>
                      {currentDataset.negative_records}
                    </Text>
                    <Text style={styles.statLbl}>Healthy Controls</Text>
                  </View>
                  <View style={styles.statBox}>
                    <Text style={[styles.statVal, { color: COLORS.primaryLight }]}>
                      {currentDataset.disease_prevalence_pct}%
                    </Text>
                    <Text style={styles.statLbl}>Prevalence</Text>
                  </View>
                </View>
              </View>
            )}

            {/* Dataset Records Table Explorer */}
            <View style={styles.explorerSection}>
              <View style={styles.tableHeaderRow}>
                <Text style={styles.sectionLabel}>Clinical Records Explorer</Text>
                <Text style={styles.recordsCountText}>
                  Showing {filteredRecords.length} records
                </Text>
              </View>

              {/* Filters Row */}
              <View style={styles.filterRow}>
                <View style={styles.searchBox}>
                  <Ionicons name="search" size={14} color={COLORS.textMuted} />
                  <TextInput
                    style={styles.searchInput}
                    placeholder="Search age, BP, chol..."
                    placeholderTextColor={COLORS.textMuted}
                    value={searchQuery}
                    onChangeText={setSearchQuery}
                  />
                  {searchQuery ? (
                    <TouchableOpacity onPress={() => setSearchQuery('')}>
                      <Ionicons name="close-circle" size={16} color={COLORS.textMuted} />
                    </TouchableOpacity>
                  ) : null}
                </View>

                <View style={styles.filterPills}>
                  <TouchableOpacity
                    style={[styles.pill, filterTarget === 'all' && styles.pillActive]}
                    onPress={() => setFilterTarget('all')}
                  >
                    <Text style={[styles.pillText, filterTarget === 'all' && styles.pillTextActive]}>
                      All
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.pill, filterTarget === 'positive' && styles.pillActivePos]}
                    onPress={() => setFilterTarget('positive')}
                  >
                    <Text style={[styles.pillText, filterTarget === 'positive' && styles.pillTextActive]}>
                      Risk (+)
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.pill, filterTarget === 'negative' && styles.pillActiveNeg]}
                    onPress={() => setFilterTarget('negative')}
                  >
                    <Text style={[styles.pillText, filterTarget === 'negative' && styles.pillTextActive]}>
                      Healthy (-)
                    </Text>
                  </TouchableOpacity>
                </View>
              </View>

              {/* Table Header */}
              <View style={styles.tableHeader}>
                <Text style={[styles.th, { width: 36 }]}>#</Text>
                <Text style={[styles.th, { width: 44 }]}>Age</Text>
                <Text style={[styles.th, { width: 40 }]}>Sex</Text>
                <Text style={[styles.th, { width: 44 }]}>CP</Text>
                <Text style={[styles.th, { width: 56 }]}>BP</Text>
                <Text style={[styles.th, { width: 56 }]}>Chol</Text>
                <Text style={[styles.th, { width: 56 }]}>MaxHR</Text>
                <Text style={[styles.th, { flex: 1, textAlign: 'right' }]}>Target</Text>
              </View>

              {/* Table Rows */}
              {filteredRecords.map((row, idx) => {
                const isDisease = row.target > 0;
                return (
                  <View
                    key={idx}
                    style={[
                      styles.tableRow,
                      idx % 2 === 1 && styles.tableRowAlt,
                    ]}
                  >
                    <Text style={[styles.td, { width: 36, color: COLORS.textMuted }]}>{idx + 1}</Text>
                    <Text style={[styles.td, { width: 44, fontWeight: '700' }]}>{row.age}</Text>
                    <Text style={[styles.td, { width: 40 }]}>{row.sex === 1 ? 'M' : 'F'}</Text>
                    <Text style={[styles.td, { width: 44 }]}>{row.cp}</Text>
                    <Text style={[styles.td, { width: 56 }]}>{row.trestbps}</Text>
                    <Text style={[styles.td, { width: 56 }]}>{row.chol}</Text>
                    <Text style={[styles.td, { width: 56 }]}>{row.thalach}</Text>
                    <View style={{ flex: 1, alignItems: 'flex-end' }}>
                      <View
                        style={[
                          styles.targetBadge,
                          isDisease ? styles.targetBadgeDisease : styles.targetBadgeHealthy,
                        ]}
                      >
                        <Text
                          style={[
                            styles.targetBadgeText,
                            { color: isDisease ? COLORS.riskHigh : COLORS.riskLow },
                          ]}
                        >
                          {isDisease ? 'Disease' : 'Healthy'}
                        </Text>
                      </View>
                    </View>
                  </View>
                );
              })}
            </View>
          </>
        )}
      </ScrollView>

      {/* Add Dataset Modal with Automated Retraining */}
      <AddDatasetModal
        visible={addModalVisible}
        onClose={() => setAddModalVisible(false)}
        onDatasetAdded={handleDatasetAdded}
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
  addBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: COLORS.primary,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 10,
  },
  addBtnText: {
    fontSize: 12,
    fontWeight: '800',
    color: COLORS.white,
  },
  telemetryBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(56, 189, 248, 0.08)',
    borderRadius: 14,
    padding: 12,
    borderWidth: 1,
    borderColor: 'rgba(56, 189, 248, 0.25)',
    marginBottom: 14,
  },
  telemetryLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    flex: 1,
  },
  telemetryTitle: {
    fontSize: 12,
    fontWeight: '800',
    color: COLORS.text,
  },
  telemetrySub: {
    fontSize: 10,
    color: COLORS.textSecondary,
    marginTop: 2,
  },
  telemetryRight: {
    alignItems: 'flex-end',
  },
  telemetryAcc: {
    fontSize: 16,
    fontWeight: '900',
    color: COLORS.riskLow,
  },
  telemetryAccLabel: {
    fontSize: 9,
    color: COLORS.textMuted,
    fontWeight: '700',
  },
  sectionLabel: {
    fontSize: 11,
    fontWeight: '700',
    color: COLORS.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 8,
  },
  hospitalTabs: {
    flexDirection: 'row',
    gap: 8,
    paddingBottom: 12,
  },
  hospTab: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: COLORS.surface,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  hospTabSelected: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primaryLight,
  },
  hospTabName: {
    fontSize: 12,
    fontWeight: '700',
    color: COLORS.text,
  },
  hospTabNameSelected: {
    color: COLORS.white,
  },
  hospTabCount: {
    fontSize: 10,
    color: COLORS.textMuted,
    marginTop: 1,
  },
  cohortCard: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    marginBottom: 16,
  },
  cohortTop: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  cohortBadgeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  hospIdBadge: {
    backgroundColor: COLORS.surfaceLight,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 6,
  },
  hospIdText: {
    fontSize: 10,
    fontWeight: '700',
    color: COLORS.primaryLight,
  },
  cohortLoc: {
    fontSize: 11,
    color: COLORS.textSecondary,
  },
  cohortTitle: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.text,
  },
  statsGrid: {
    flexDirection: 'row',
    gap: 8,
  },
  statBox: {
    flex: 1,
    backgroundColor: COLORS.surface,
    padding: 8,
    borderRadius: 10,
    alignItems: 'center',
  },
  statVal: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.text,
  },
  statLbl: {
    fontSize: 9,
    color: COLORS.textMuted,
    marginTop: 2,
    textAlign: 'center',
  },
  explorerSection: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  tableHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  recordsCountText: {
    fontSize: 11,
    color: COLORS.textMuted,
  },
  filterRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 12,
  },
  searchBox: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: COLORS.inputBg,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    paddingHorizontal: 8,
    height: 34,
  },
  searchInput: {
    flex: 1,
    fontSize: 11,
    color: COLORS.text,
  },
  filterPills: {
    flexDirection: 'row',
    gap: 4,
  },
  pill: {
    paddingHorizontal: 8,
    paddingVertical: 6,
    borderRadius: 8,
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  pillActive: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primaryLight,
  },
  pillActivePos: {
    backgroundColor: 'rgba(239, 68, 68, 0.25)',
    borderColor: COLORS.riskHigh,
  },
  pillActiveNeg: {
    backgroundColor: 'rgba(16, 185, 129, 0.25)',
    borderColor: COLORS.riskLow,
  },
  pillText: {
    fontSize: 10,
    fontWeight: '600',
    color: COLORS.textSecondary,
  },
  pillTextActive: {
    color: COLORS.white,
    fontWeight: '700',
  },
  tableHeader: {
    flexDirection: 'row',
    backgroundColor: COLORS.surface,
    paddingVertical: 8,
    paddingHorizontal: 6,
    borderRadius: 6,
    marginBottom: 4,
  },
  th: {
    fontSize: 10,
    fontWeight: '800',
    color: COLORS.textSecondary,
    textTransform: 'uppercase',
  },
  tableRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 7,
    paddingHorizontal: 6,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.04)',
  },
  tableRowAlt: {
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
  },
  td: {
    fontSize: 11,
    color: COLORS.text,
  },
  targetBadge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 6,
  },
  targetBadgeDisease: {
    backgroundColor: 'rgba(239, 68, 68, 0.15)',
  },
  targetBadgeHealthy: {
    backgroundColor: 'rgba(16, 185, 129, 0.15)',
  },
  targetBadgeText: {
    fontSize: 9,
    fontWeight: '800',
  },
  loadingWrap: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  loadingText: {
    fontSize: 13,
    color: COLORS.textSecondary,
    marginTop: 10,
  },
});
