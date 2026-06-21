import { router, useLocalSearchParams } from 'expo-router';
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import theme from '../constants/theme';

export default function ResultScreen() {
  const {
    imageUri,
    label,
    resultTitle,
    fakeProbability,
    baselineProbability,
    detailProbability,
    summary,
    reasons,
    llmExplanation,
  } = useLocalSearchParams<{
    imageUri: string;
    label: string;
    resultTitle: string;
    fakeProbability: string;
    baselineProbability: string;
    detailProbability: string;
    summary: string;
    reasons: string;
    llmExplanation: string;
  }>();

  const isFake = label === 'fake';
  const resultText = isFake ? 'FAKE' : 'REAL';

  const percent = (value?: string) => {
    if (!value) return '0.0%';
    return `${Number(value).toFixed(1)}%`;
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>분석 결과</Text>

      <View style={styles.imageBox}>
        <Image source={{ uri: imageUri }} style={styles.image} />

        {isFake && (
          <View style={styles.fakeBox}>
            <Text style={styles.fakeBoxText}>Suspicious</Text>
          </View>
        )}
      </View>

      <View style={styles.resultCard}>
        <Text style={[styles.resultText, { color: isFake ? theme.danger : theme.success }]}>
          {resultText}
        </Text>

        <Text style={styles.probText}>딥페이크 확률</Text>
        <Text style={styles.percent}>{percent(fakeProbability)}</Text>

        <View style={styles.line} />

        <Text style={styles.desc}>
          {summary || 'AI 분석 요약을 불러오지 못했습니다.'}
        </Text>
      </View>

      <Pressable
        style={styles.btn}
        onPress={() =>
          router.push({
            pathname: '/report',
            params: {
              imageUri,
              label,
              resultTitle,
              fakeProbability,
              baselineProbability,
              detailProbability,
              summary,
              reasons,
              llmExplanation,
            },
          })
        }
      >
        <Text style={styles.btnText}>AI 리포트 보기</Text>
      </Pressable>

      <Pressable style={styles.subBtn} onPress={() => router.replace('/')}>
        <Text style={styles.subBtnText}>처음으로</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.bg,
  },
  content: {
    padding: 22,
    paddingBottom: 40,
  },
  title: {
    color: theme.text,
    fontSize: 27,
    fontWeight: '800',
    marginTop: 10,
    marginBottom: 20,
  },
  imageBox: {
    width: '100%',
    height: 310,
    borderRadius: 24,
    backgroundColor: theme.card,
    overflow: 'hidden',
    position: 'relative',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  fakeBox: {
    position: 'absolute',
    left: '34%',
    top: '45%',
    width: 130,
    height: 80,
    borderWidth: 2,
    borderColor: theme.danger,
    backgroundColor: 'rgba(255,77,109,0.18)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  fakeBoxText: {
    color: theme.text,
    fontSize: 12,
    fontWeight: '700',
  },
  resultCard: {
    backgroundColor: theme.card,
    borderRadius: 24,
    padding: 24,
    borderWidth: 1,
    borderColor: theme.border,
    marginTop: 24,
  },
  resultText: {
    fontSize: 38,
    fontWeight: '900',
  },
  probText: {
    color: theme.subText,
    fontSize: 14,
    marginTop: 14,
  },
  percent: {
    color: theme.text,
    fontSize: 32,
    fontWeight: '800',
    marginTop: 4,
  },
  line: {
    height: 1,
    backgroundColor: theme.border,
    marginVertical: 18,
  },
  desc: {
    color: theme.subText,
    fontSize: 15,
    lineHeight: 23,
  },
  btn: {
    backgroundColor: theme.primary,
    paddingVertical: 17,
    borderRadius: 18,
    alignItems: 'center',
    marginTop: 24,
  },
  btnText: {
    color: theme.text,
    fontSize: 16,
    fontWeight: '700',
  },
  subBtn: {
    alignItems: 'center',
    paddingVertical: 16,
  },
  subBtnText: {
    color: theme.subText,
    fontSize: 14,
  },
});