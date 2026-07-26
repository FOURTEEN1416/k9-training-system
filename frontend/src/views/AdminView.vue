<script setup lang="ts">
import { NCard, NSpace, NText, NDescriptions, NDescriptionsItem, NTag } from 'naive-ui'
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()
</script>

<template>
  <NSpace vertical :size="16">
    <NCard title="系统信息" size="medium" bordered>
      <NDescriptions :column="2" label-placement="left" bordered>
        <NDescriptionsItem label="系统名称">工作犬训练机器视觉识别系统</NDescriptionsItem>
        <NDescriptionsItem label="当前阶段">
          <NTag type="info" size="small">Phase 0 基础设施</NTag>
        </NDescriptionsItem>
        <NDescriptionsItem label="后端状态">
          <NTag :type="appStore.backendOnline ? 'success' : 'error'" size="small">
            {{ appStore.backendOnline ? '在线' : '离线' }}
          </NTag>
        </NDescriptionsItem>
        <NDescriptionsItem label="后端版本">
          {{ appStore.health?.version || '—' }}
        </NDescriptionsItem>
        <NDescriptionsItem label="运行环境">
          {{ appStore.health?.environment || '—' }}
        </NDescriptionsItem>
        <NDescriptionsItem label="API 文档">
          <a href="/docs" target="_blank">Swagger UI</a>
        </NDescriptionsItem>
      </NDescriptions>
    </NCard>

    <NCard title="模型管理" size="medium" bordered>
      <template #header-extra>
        <NText depth="3" style="font-size: 12px;">Phase 1 实现</NText>
      </template>
      <NText depth="3">YOLO26-pose / PoseC3D / ST-GCN+BC 模型注册与激活将在 Phase 1 启用</NText>
    </NCard>

    <NCard title="用户与权限" size="medium" bordered>
      <template #header-extra>
        <NText depth="3" style="font-size: 12px;">Phase 2 实现</NText>
      </template>
      <NText depth="3">训导员 / 管理员 / 研究员 角色与权限将在 Phase 2 启用</NText>
    </NCard>
  </NSpace>
</template>
