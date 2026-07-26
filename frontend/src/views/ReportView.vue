<script setup lang="ts">
import { NCard, NSpace, NText, NEmpty, NStatistic } from 'naive-ui'
import { useRoute } from 'vue-router'
import { computed } from 'vue'

const route = useRoute()
const reportId = computed(() => route.params.id as string | undefined)
</script>

<template>
  <NSpace vertical :size="16">
    <NCard title="评分报告" size="medium" bordered>
      <template #header-extra>
        <NText depth="3" style="font-size: 12px;">Phase 1 实现</NText>
      </template>
      <NSpace vertical :size="12">
        <NText v-if="reportId">报告 ID：{{ reportId }}</NText>
        <NText v-else>未指定报告 ID（请从历史记录页进入）</NText>
      </NSpace>
    </NCard>

    <NCard title="7 维评分" size="medium" bordered>
      <NSpace :size="24" wrap>
        <NStatistic label="准确度" :value="'—'" />
        <NStatistic label="响应延迟" :value="'—'" />
        <NStatistic label="保持时长" :value="'—'" />
        <NStatistic label="搜索效率" :value="'—'" />
        <NStatistic label="注意力" :value="'—'" />
        <NStatistic label="胆量欲望" :value="'—'" />
        <NStatistic label="步态质量" :value="'—'" />
      </NSpace>
    </NCard>

    <NCard title="行为识别结果" size="medium" bordered>
      <NEmpty description="Phase 1 将展示 22 种行为分类的时间轴" />
    </NCard>

    <NCard title="姿态关键点" size="medium" bordered>
      <NEmpty description="Phase 1 将展示 24 个关键点的可视化" />
    </NCard>
  </NSpace>
</template>
