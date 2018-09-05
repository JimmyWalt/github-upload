package com.jiy.nlp

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.sql.SQLContext
import java.util.Properties
import org.apache.spark.sql.DataFrame
import org.apache.commons.io.IOUtils
import java.io.FileInputStream
import java.io.File
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.feature.Word2Vec
//import com.jiy.nlp.Segmentor
import java.util.Date
import java.text.SimpleDateFormat
import org.apache.hadoop.fs.{Path,FileSystem}
import java.io.FileOutputStream
import java.io.ObjectOutputStream
import java.io.ObjectInputStream
import java.util.Calendar

object smplus {
    //配置文件加载，输入参数控制文件
    var config = new Properties()
    val configFile = "conf/sm.properties"
    config.load(new FileInputStream(new File(configFile)))
  
    /**语料。*/
	  var corpusSentence:RDD[List[String]]=null
	  /**语料训练模型。*/
	  var model: Word2VecModel=null
    
  def main(args: Array[String]): Unit = {
    
    val conf = new SparkConf().setAppName(this.getClass.toString()).setMaster("local[4]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    
    val isInside=config.getProperty("isInside").trim().toBoolean //  false   是否内网
    val w2vTraining=config.getProperty("w2vTraining").trim().toBoolean //false   是否训练模型
    val localMode=config.getProperty("localMode").trim().toBoolean  //true    是否本地存取模型
    val islocal=config.getProperty("islocal").trim().toBoolean    //true     是否mysql读取topic
    val inputTopic=if(islocal) "result" else config.getProperty("inputTopic").trim()  
    val savePath = config.getProperty("savePath").trim()+inputTopic+".csv"
    val istoday = config.getProperty("istoday").trim().toBoolean
    val firstLayer=config.getProperty("firstLayer").trim().toInt
    val secondLayer=config.getProperty("secondLayer").trim().toInt
    val today = if(istoday) NowDate()   else    "2018-07-15"
    val pastday = pastDate(today, 10) 
    
    //mysql ip地址 用户名密码 及库名
    val url=if(isInside) "jdbc:mysql://172.16.210.83:3306/news_db"
                   else  "jdbc:mysql://47.93.193.31:3306/news_db"
//                  else "jdbc:mysql://124.202.155.72:33063/news_db"
    val username="newsreadonly"
    val pwd="newsreadonly"
    val uri=url+"?user="+username+"&password="+pwd+"&useUnicode=true&characterEncoding=UTF-8"
 
    //mysql  jdbc参数设置
    val prop = new Properties()
    
//    val sqlstr="SELECT content FROM news_db.t_news_detail"
    val sqlstr=s"select id,title,content from news_db.t_news_detail where DATE_FORMAT(create_time,'%Y-%m-%d') between ${today} and ${pastday};"
    
    import sqlContext.implicits._
    prop.put("driver","com.mysql.jdbc.Driver")
    //文章库

//    val df1 = sqlContext.read.jdbc(uri,"t_news_detail","id",4120000,4128693,10,prop )
//    val df1 = sqlContext.read.jdbc(uri,"t_news_detail","id",3846637,3946637,10,prop )
//    val dfs00 = sqlContext.read.jdbc(uri,"t_news_detail",prop )
    //topic库
    val topiclow=config.getProperty("topiclow").trim().toLong
    val topicup=config.getProperty("topicup").trim().toLong
    val contentlow=config.getProperty("contentlow").trim().toLong
    val contentup=config.getProperty("contentup").trim().toLong
    val predicates1 = Array[String](s"id >= ${topiclow} and id <= ${topicup} ")
    val df2 = sqlContext.read.jdbc(uri,"t_news_topic",prop)
//    val df2 = sqlContext.read.jdbc(uri,"t_news_topic",predicates1,prop)
//    val df2 = sqlContext.read.jdbc(uri,"t_news_topic","id",2724000L,2724634L,10,prop)
//    val df2 = sqlContext.read.jdbc(uri,"t_news_topic","id",872341,982341,10,prop)
//    SELECT `create_time`,`name` FROM t_news_topic WHERE id < 973341
    
//    sqlContext.setConf(prop)
//    sqlContext.sql(sqlstr)
    
    var contentrdd:DataFrame = null
    //word2vec路径
    val w2vModelFile = config.getProperty("w2vModelFile").trim
    //词典训练数据路径
    val TrainPath=config.getProperty("TrainPath").trim
    
    if(w2vTraining) {
        val predicates = Array[String](s"id >= ${contentlow} and id <= ${contentup} ")
        val df1 = sqlContext.read.jdbc(uri,"t_news_detail",predicates,prop )
        contentrdd = df1.select("id","title","content")//.where($"id"<=1748324)
	  		training(sc,TrainPath,contentrdd)
		  	if(localMode) saveModelLocal(w2vModelFile) else saveModel(sc,w2vModelFile)
	  	}else{	
        if(localMode) loadModelLocal(w2vModelFile) else loadModel(sc,w2vModelFile)
	  	}
    
    //word2vec模型 词向量
    val mapx = model.getVectors.toArray.zipWithIndex//.toArray 
    val dicset = mapx.map(_._1._1)
    val dicmap = mapx.map(e=>(e._1._1,e._2)).toMap
    
    //提取topic
    val topicrdd =if(islocal) {
                      val tmp= df2.select("create_time","name")//.limit(10)//.filter("DATE(create_time)='20180727'")  
                      tmp.map{e=> (e(0).toString().slice(0,10),e(1).toString().split("#",-1)(0))}
                      .filter(e=>e._1==today)
                      .map(elem=>(elem._2,1)).rdd.reduceByKey(_+_).map(elem=>(elem._1,elem._2.toDouble)) 
                      .filter(elem=>dicset.contains(elem._1))}
                 else {
                   val word = (inputTopic.split(",",-1)).map((_,0.3))   
                     sc.parallelize(word)
                 }
    
    //第一层相似词提取
    val result1 = topicrdd.flatMap{case(word,t)=>
        val tmp=model.findSynonyms(word,firstLayer)
        tmp.map(e=>(word,e._1))
    }  
    val result2=(result1.map(e=>(e._2,1))).reduceByKey(_+_)
    //第二层相似词提取
    val result3=result2.flatMap{case(word,t)=>
       val tmp =  model.findSynonyms(word,secondLayer)
       tmp.map(e=>(word,e._1))
    }
    val result4=(result3++result1).map{e=>
       Array(dicmap.getOrElse(e._1,-1).toString(),e._1,dicmap.getOrElse(e._2,-1).toString(),e._2) 
    }.map(_.mkString(",")) 
    
    //文件保存
    val path = new Path(savePath);
    val path1= new Path(savePath+"topic")
    val hadoopConf = sc.hadoopConfiguration
    val hdfs = FileSystem.get(hadoopConf)
    if(hdfs.exists(path)){
      hdfs.delete(path,true)
    }
    if(hdfs.exists(path1)){
      hdfs.delete(path1,true)
    }
    result4.coalesce(1).saveAsTextFile(savePath)
    topicrdd.coalesce(1).saveAsTextFile(savePath+"topic")
    
//    sc.stop()
  }
  
  
  def getProperty(property:String)={
	  config.getProperty(property).trim()
	  IOUtils.toString(config.getProperty(property).trim().toArray.map { x => x.toByte },"UTF-8")
	}
  
  def test(sqlContext:SQLContext)={
    val documentDF = sqlContext.createDataFrame(Seq(
        "Hi I heard about Spark".split(" "),
        "I wish Java could use case classes".split(" "),
        "Logistic regression models are neat".split(" ")
        ).map(Tuple1.apply)).toDF("text")
  }
  
  def training(sc:SparkContext,corpusPath:String,df:DataFrame)={
	  val dfx=df.rdd.map(elem=>elem(2).toString())
	  var corpus = dfx.filter(x => x.take(1).matches("[\u4E00-\u9FCC]")).cache()
//	  var corpusCrowler = sc.wholeTextFiles(corpusPath+"/*/*").map(x=>x._2).filter(x => x.take(1).matches("[\u4E00-\u9FCC]")).cache()
//		var corpusParallel = sc.textFile(corpusPath+"/corpus-50W.txt").filter(x => x.take(1).matches("[\u4E00-\u9FCC]")).cache()
//		val corpusWord=corpusParallel.filter(x => x.length()<=5).collect().toList
//		corpus = corpusCrowler++corpusParallel++corpus
		corpusSentence=corpus.filter(x => x.length()>5 ).map(t => {
			Segmentor.parseTuple(t).map { x => x._1 }.toList
		})
//		corpusSentence=corpusSentence++SparkSC.sc.parallelize(List(corpusWord))
		val word2vec = new Word2Vec()
		model = word2vec.fit(corpusSentence)
		
	}
  
  def NowDate(): String = {
    val now: Date = new Date()
    
    val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd")
    val date = dateFormat.format(now)
    date
  }
  
  def pastDate(getEndDate:String,spanDay:Int)={
    
    val formatT = new SimpleDateFormat("yyyy-MM-dd") 
    val endDate = formatT.parse(getEndDate)
    val calendarPerdict = Calendar.getInstance()
    calendarPerdict.setTime(endDate)
    calendarPerdict.add(Calendar.DATE, -spanDay)
    val perdictDate = calendarPerdict.getTime
    val perdictDateStr = formatT.format(perdictDate)
    perdictDateStr
  }
  
  def saveModelLocal(path:String)={
		val fos = new FileOutputStream(path) 
		val oos = new ObjectOutputStream(fos)   
		oos.writeObject(model)   
		oos.close
	}
  
  def saveModel(sc:SparkContext,path:String)={
		model.save(sc, path)
	}
  
  def loadModelLocal(path:String)={
		model=null
		val fos = new FileInputStream(path) 
		val oos = new ObjectInputStream(fos) 
		model= oos.readObject().asInstanceOf[Word2VecModel]
	}
  
  def loadModel(sc:SparkContext,path:String)={
		model=null
		model=Word2VecModel.load(sc, path)
	}
 
}