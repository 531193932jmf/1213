  *	?????7?@2u
>Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::BatchV2?i?q??@!?w?1?X@)???~?:??1E|?y?J@:Preprocessing2?
eIterator::Model::MaxIntraOpParallelism::ForeverRepeat::BatchV2::Shuffle::MemoryCacheImpl::TensorSlice?H?}8g??!b/??R ?@)H?}8g??1b/??R ?@:Preprocessing2
GIterator::Model::MaxIntraOpParallelism::ForeverRepeat::BatchV2::Shuffle???+e???!YrE??F@)>yX?5???12??v?'@:Preprocessing2?
XIterator::Model::MaxIntraOpParallelism::ForeverRepeat::BatchV2::Shuffle::MemoryCacheImpl?H?}8g??!??? ?
B@)      ??1ˏ??vS@:Preprocessing2?
TIterator::Model::MaxIntraOpParallelism::ForeverRepeat::BatchV2::Shuffle::MemoryCache?7?A`????!s??v??C@)vOjM??1???[u?@:Preprocessing2F
Iterator::Modelvq??@!      Y@) ?o_?y?15?ڰd??:Preprocessing2l
5Iterator::Model::MaxIntraOpParallelism::ForeverRepeat??j+??@!??V)?X@)HP?s?r?1r?GC????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??k	??@!	%O???X@)/n??r?1ܔ?l.???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.