  *	?????ì@2?
eIterator::Model::MaxIntraOpParallelism::ForeverRepeat::BatchV2::Shuffle::MemoryCacheImpl::TensorSlice????(???!E?G??G@)???(???1E?G??G@:Preprocessing2u
>Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::BatchV2?&?W@!)3? ??X@)?X?? ??1??sI??A@:Preprocessing2?
XIterator::Model::MaxIntraOpParallelism::ForeverRepeat::BatchV2::Shuffle::MemoryCacheImpl?jM?@!?*z??L@)4??7????1?y?(S%@:Preprocessing2
GIterator::Model::MaxIntraOpParallelism::ForeverRepeat::BatchV2::Shuffle?z6?>?@!?}~???O@)5^?I??1?9?Z'"@:Preprocessing2?
TIterator::Model::MaxIntraOpParallelism::ForeverRepeat::BatchV2::Shuffle::MemoryCache?,Ԛ??@!?%?`?M@)??&S??1ﾮZ?$ @:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Q??k@!;?L???X@)Ǻ???v?1ߣ8?w??:Preprocessing2F
Iterator::ModeljM??St@!      Y@)"??u??q?1}??????:Preprocessing2l
5Iterator::Model::MaxIntraOpParallelism::ForeverRepeat?uq`@!K8????X@)	?^)?p?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.