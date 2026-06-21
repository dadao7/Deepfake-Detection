import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, View } from 'react-native';
import theme from '../constants/theme';

const BASE_URL = 'https://seallike-shalonda-nonsynchronous.ngrok-free.dev';

export default function LoadingScreen() {
  const { imageUri, jobId } = useLocalSearchParams<{
    imageUri: string;
    jobId: string;
  }>();

  const [step, setStep] = useState(0);

  const steps = [
    '이미지 업로드 완료',
    '얼굴 영역 탐지 중',
    '시각 특징 추출 중',
    '위조 흔적 분석 중',
    'Detail Region 분석 중',
    '최종 결과 정리 중',
    'AI 리포트 생성 중',
  ];

  useEffect(() => {
    if (!jobId) {
      Alert.alert('오류', '분석 작업 ID가 없습니다.');
      router.back();
      return;
    }

    const stepInterval = setInterval(() => {
      setStep((prev) => {
        if (prev < steps.length - 1) return prev + 1;
        return prev;
      });
    }, 100000);

    const resultInterval = setInterval(async () => {
      try {
        const response = await fetch(`${BASE_URL}/result/${jobId}`);
        const data = await response.json();

        console.log('result polling:', data);

        if (data.status === 'done') {
          clearInterval(stepInterval);
          clearInterval(resultInterval);

          router.replace({
            pathname: '/result',
            params: {
              imageUri,
              label: data.label,
              resultTitle: data.result_title,
              resultColor: data.result_color,
              fakeProbability: String(data.fake_probability),
              baselineProbability: String(data.baseline_fake_probability),
              detailProbability: String(data.detail_fake_probability),
              summary: data.summary,
              reasons: JSON.stringify(data.reasons ?? []),
              llmExplanation: data.llm_explanation ?? '',
            },
          });
        }

        if (data.status === 'failed') {
          clearInterval(stepInterval);
          clearInterval(resultInterval);

          console.log(data.error);
          Alert.alert('분석 실패', '모델 분석 중 오류가 발생했습니다.');
          router.back();
        }
      } catch (error) {
        console.log(error);
      }
    }, 60000);

    return () => {
      clearInterval(stepInterval);
      clearInterval(resultInterval);
    };
  }, [jobId]);

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color={theme.accent} />

      <Text style={styles.title}>AI 분석 중</Text>
      <Text style={styles.step}>{steps[step]}</Text>

      <View style={styles.progressBox}>
        {steps.map((item, index) => (
          <View key={item} style={styles.row}>
            <View style={[styles.dot, index <= step && styles.activeDot]} />
            <Text style={[styles.rowText, index <= step && styles.activeText]}>
              {item}
            </Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.bg,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  title: {
    color: theme.text,
    fontSize: 30,
    fontWeight: '800',
    marginTop: 30,
  },
  step: {
    color: theme.accent,
    fontSize: 16,
    marginTop: 12,
    marginBottom: 40,
  },
  progressBox: {
    width: '100%',
    backgroundColor: theme.card,
    borderRadius: 24,
    padding: 22,
    borderWidth: 1,
    borderColor: theme.border,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 18,
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#333B55',
    marginRight: 12,
  },
  activeDot: {
    backgroundColor: theme.accent,
  },
  rowText: {
    color: theme.subText,
    fontSize: 15,
  },
  activeText: {
    color: theme.text,
    fontWeight: '700',
  },
});