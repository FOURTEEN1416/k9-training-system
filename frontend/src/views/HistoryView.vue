<script setup lang="ts">
import { NCard, NSpace, NText, NDataTable, NEmpty, type DataTableColumns } from 'naive-ui'
import type { Video } from '@/api'

const columns: DataTableColumns<Video> = [
  { title: 'ID', key: 'id', width: 80 },
  { title: '文件名', key: 'original_filename' },
  { title: '犬只', key: 'dog_id' },
  { title: '状态', key: 'status' },
  { title: '上传时间', key: 'uploaded_at' },
]

const data: Video[] = []
</script>

<template>
  <NSpace vertical :size="16">
    <NCard title="历史记录" size="medium" bordered>
      <template #header-extra>
        <NText depth="3" style="font-size: 12px;">Phase 1 实现</NText>
      </template>
      <NDataTable
        :columns="columns"
        :data="data"
        :bordered="false"
        :pagination="{ pageSize: 20 }"
      >
        <template #empty>
          <NEmpty description="暂无视频记录（Phase 1 启用上传后将显示）" />
        </template>
      </NDataTable>
    </NCard>
  </NSpace>
</template>
