package com.jiy.nlp

import scala.collection._
import scala.collection.mutable.ListBuffer
import scala.collection.Iterable._
import scala.collection.TraversableOnce._
import scala.annotation._
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import scala.sys._
import scala.util.control._
import scala.util._
import scala.util.matching._
import scala.concurrent._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.control.Breaks._
import scala.xml._

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.storage.StorageLevel

import java.io.File
import java.util.Map.Entry
import java.util.HashMap
import org.slf4j._
import org.apache.commons.io._

import org.ansj.splitWord.analysis._
import org.ansj.domain.Term
import org.ansj.util.MyStaticValue
import org.ansj.library.UserDefineLibrary
//import org.ansj.library.DicLibrary
import org.ansj.dic._
import org.ansj.app.keyword._

import org.ansj.recognition.NatureRecognition


/**
 * 中文分词器。
 */
case class ZhSegmentor(var dicPath:String=System.getProperty("user.dir")+"/conf/dictionary/",var dicSeg:String="segment") extends Serializable{
  /**默认的分词类型：Nlp分词，To精准分词，Base基本分词。*/
	var segType="Nlp"  //Nlp,To
	var learnTool = new LearnTool()
	//var dicPath=""
	//var dicSeg=""
	var userDic=List[String]()
	init(dicPath,dicSeg)
	
	def setPath(dicPth:String,dicSegPth:String)={
	 	dicPath=dicPth
  	dicSeg=dicSegPth
  	loadUserLibrary(dicPath+dicSeg)
  	loadAmbiguityLibrary(dicPath+"ambiguity.dic") 
	}
	
	def init(dicPth:String,dicSegPth:String)={
	  setPath(dicPth,dicSegPth)
	  val dicFiles=new File(dicPth+dicSegPth+"/newwords.dic")
	  //println("dicFiles="+dicFiles+","+dicFiles.isFile)
	  //println("dicPath+dicSeg="+dicPath+dicSeg)
  	if(dicFiles.isFile){
  		userDic=FileUtils.readLines(dicFiles).filter { x => x.length()>0&&(x.contains("\t")) }.map { x => x.split("\t")(0).trim }.toList
  	}
  	//println("userDic="+userDic.take(100).mkString("\n"))
	  println(parseTuple("中文自然语言分词器。").mkString(" | "))
	}
	
	def loadDicLibrary(dicPath:String,dicSeg:String)={
	  loadUserLibrary(dicPath+dicSeg)
  	loadAmbiguityLibrary(dicPath+"ambiguity.dic")
	}
	
	def getSegType()=segType
	
	def setSegType(segT:String)={
		segType=segT
	}
	
	def loadUserLibrary(dicPth:String)={
		MyStaticValue.userLibrary=dicPth
//		MyStaticValue.ENV.put(DicLibrary.DEFAULT, dicPth) 
	}
	def loadAmbiguityLibrary(dicPth:String)={
		MyStaticValue.ambiguityLibrary=dicPth
//		MyStaticValue.ENV.put(, value)
	}
	
	def getFilesList(path:String):scala.List[String]={
		var inputFiles=ListBuffer[String]()
		val fl = new File(path)
		println("fl.list(path)="+path)
		println("fl.list()="+fl.list())
		val fs = fl.list().toList
		println("fs="+fs.mkString(","))
		for (x <- fs) {
			if (!x.startsWith(".")){
				val opFile = new File(path + x)
				if(opFile.isFile()){
					inputFiles+=(path+x)
				}else if(opFile.isDirectory()){
					val opFs = opFile.list()
					for(y<-opFs){
						inputFiles+= (path+x+"/"+y)
					}
				}
			}
		}
		inputFiles.toList
	}
	
	def addDicLibrary(dicPth:String)={
		val dic=FileUtils.readLines(new File(dicPth))
		dic.foreach( x => {
			val line=x.split("\t")
			if(line.length>2) {
				UserDefineLibrary.insertWord(line(0).trim, line(1).trim, line(2).trim.toInt)
			}else if(line.length==1 || line.length==2){
				UserDefineLibrary.insertWord(line(0).trim, "userDefine", 10)
			}
		})
	}
	
	def parse(text:String,learn:Boolean=false):List[Term]={
		segType match{
			case "Base" => return BaseAnalysis.parse(text.replaceAll("[\\s\u3000]","")).toList
			case "To" => return ToAnalysis.parse(text.replaceAll("[\\s\u3000]","")).toList
			case "Nlp" => if(!learn) return NlpAnalysis.parse(text.replaceAll("[\\s\u3000]","")).toList else NlpAnalysis.parse(text.replaceAll("[\\s\u3000]",""),learnTool).toList
			case _ => List[Term]()
		}
	}
	
	def parseTuple(text:String,learn:Boolean=false):List[(String,String)]={
		segType match{
			case "Base" => return BaseAnalysis.parse(text.replaceAll("[\\s\u3000]","")).map { x => (x.getName.trim,x.getNatureStr.trim) }.toList
			case "To" => return ToAnalysis.parse(text.replaceAll("[\\s\u3000]","")).map { x => (x.getName.trim,x.getNatureStr.trim) }.toList
			case "Nlp" => if(!learn) return NlpAnalysis.parse(text.replaceAll("[\\s\u3000]","")).map { x => (x.getName.trim,x.getNatureStr.trim) }.toList else NlpAnalysis.parse(text.replaceAll("[\\s\u3000]",""),learnTool).map { x => (x.getName.trim,x.getNatureStr.trim) }.toList
			case _ => List[(String,String)]()
		}
	}
	
	def newWord(text:String,top:Int):List[(String,Double)]={
		learnTool=null
		learnTool=new LearnTool()
		//words=null
		var words=List[Term]()
		words=parse(text.replaceAll("[\\s\u3000]",""),true)
		val topNewWords=learnTool.getTopTree(top)
		if(topNewWords!=null&&topNewWords.length>0) topNewWords.map(x=> (x.getKey,x.getValue.toDouble)).filter(w => !(w._1.matches("[\\s\u3000]")) ).toList else List[(String,Double)](("",0.0))
	}
	
	def getKeyWord(text:String,top:Int):List[(String,Double,Double)]={
		val kw=new KeyWordComputer(top).computeArticleTfidf(text.replaceAll("[\\s\u3000]",""))
		if(kw!=null&&kw.nonEmpty) {
			val keys=kw.map { x => (x.getName,x.getScore,x.getFreq.toDouble) }.toList
			var (sum1,sum2) = keys.map { w => (w._2,w._3) }.reduce((u,v)=>(u._1+v._1,u._2+v._2))
			if(sum1<0.1) sum1=1.0
			if(sum2<0.1) sum2=1.0
			keys.map( w=>(w._1, w._2/sum1, w._3/sum2) ).toList
		} else List[(String,Double,Double)]()
		//new KeyWordComputer(top).computeArticleTfidf(text.replaceAll("[\\s\u3000]","")).map { x => (x.getName,x.getScore,x.getFreq.toDouble) }.toList
	}
	
	/** 找出文本的新词。
	 *  @param text String，文本。
	 *  @param top Int，文本新词个数。
	 *  @return java.util.List<java.util.Map.Entry<String,java.lang.Double>>，新词列表。Entry(String,Double)表示新词和新词分数。
	 */
	def newWordJava(text:String,top:Int):java.util.List[java.util.Map.Entry[String,java.lang.Double]]={
		//learnTool=null
		learnTool=new LearnTool()
		//words=null
		var words=List[Term]()
		words=parse(text.replaceAll("[\\s\u3000]",""),true)
		val topNewWords=learnTool.getTopTree(top)
		learnTool=null
		val hmap=new HashMap[String, java.lang.Double]()
		hmap.put("无", 0.0)
		if(topNewWords!=null&&topNewWords.nonEmpty) {
			topNewWords.filter(x=>{
				val s=x.getKey
				!s.matches("[\\s\u3000]+") && s.length()>0 && (!userDic.contains(s))
			}) 
		}else{
			new java.util.ArrayList[java.util.Map.Entry[String,java.lang.Double]](hmap.entrySet().take(1))
		}
	}
		    
	def parseJava(text:String,learn:Boolean=false):java.util.List[String]={
		segType match{
			case "Base" => return BaseAnalysis.parse(text.replaceAll("[\\s\u3000]","")).map { x => x.getName }.toList.asJava
			case "To" => return ToAnalysis.parse(text.replaceAll("[\\s\u3000]","")).map { x => x.getName }.toList.asJava
			case "Nlp" => if(!learn) return NlpAnalysis.parse(text.replaceAll("[\\s\u3000]","")).map { x => x.getName }.toList.asJava else NlpAnalysis.parse(text.replaceAll("[\\s\u3000]",""),learnTool).map { x => x.getName }.toList.asJava
			case _ => List[String]().asJava
		}
	}
}

object SegTest{
	def main(args:Array[String])={
		//Segmentor.setSegType("Nlp")
	  //Segmentor.setSegType("Base")
		//val w=Segmentor.newWordJava("可以打着这个需求点去运作的互联网公司不应只是社交类软件与可穿戴设备，还有携程网，去哪儿网等等，订房订酒店多好的寓意。",10)
		//println(w)
		//println(w.get(0)._1+"|"+w.get(0)._2)
		//Segmentor.parseTuple("阵痛期,可以打着这个需求点去运作的互联网公司不应只是社交类软件与可穿戴设备，还有携程网，去哪儿网等等，订房订酒店多好的寓意。").foreach(println)
		/*Segmentor.parseTuple("""造屋要架梁，撒网要抓纲。适应和引领新常态，必须抓住关键问题，拎住“衣领子”、牵住“牛鼻子”。当前，制约我国经济发展的因素，供给和需求两侧都有，但矛盾的主要方面在供给侧。
		  比如，产能过剩已经成为一大顽疾，如果这个结构性矛盾得不到解决，工业品价格就会持续下降，企业效益就不可能提升，经济增长也就难以持续。
		  进入发展新阶段，放水漫灌强刺激的事情不能再干了，投资没回报、产品没市场的项目不能再上了，抱着粗放型发展方式老黄历不放，必然会走进死胡同。
		  只有用改革的办法推进结构调整，用创新的力量打造动力引擎，减少无效和低端供给，扩大有效和中高端供给，才能让供给侧凤凰涅槃，不断培育发展新动力、厚植发展新优势。
		  从这个意义上说，推进供给侧结构性改革，是适应和引领经济发展新常态的重大创新，也是适应我国经济发展新常态的必然要求。""").foreach(println)*/
		/*Segmentor.parseTuple("""
		台湾民进党“立委”高志鹏近日提出3项修法提案，主张废除学校、公务单位悬挂国父遗像的规定，未来“总统”就职将不再向国父遗像行礼。这项提案预计26日将在“立法院”闯关。
		高志鹏是陈水扁嫡系人马，1993年起开始加入陈水扁“立委”办公室，2002年起担任台湾“立委”。他在上届“立法院”曾提案修改“国徽国旗法”“宣誓条例”等等，但遭国民党团阻挡，无法进入审查程序。此次他卷土重来，由于民进党已在“立法院”掌握多数席位，如果民进党集体支持，该提案就将通过。
		此事在台湾激起轩然大波，蓝营人士及媒体痛斥高的提案，要求蔡英文表态，蔡于昨天做出态度模糊的回应，不置可否。而《自由时报》等绿营力量则嘲笑孙中山是“只挂在墙壁上的国父”，并且宣扬孙中山几次来台都与台湾的命运实际无关。台《联合报》质疑民进党改走“装饰主义台独”，认为民进党如今不敢真枪实弹迈向“建国之路”，只好忸忸怩怩改走曲径幽巷，因而选举之后“转型正义”如雨后春笋冒出头。
		今年11月12日是孙中山先生诞辰150周年，大陆方面已宣布将举行隆重纪念活动，而台湾民进党却有可能将胜选后的第一刀砍向这位中国民主革命的伟大先驱。
		蔡英文胜选前后多次宣称将致力于“维持现状”，然而一旦孙中山的国父地位从台湾正式场合消失，就意味着释放出将用切香肠方式把她所承诺维护的现状一步步搞没的信号。
		尊崇国父是“中华民国”延续了70几年的传统，它如今是台湾公开标志自己“中国血统”为数不多的政治符号之一。孙中山还长期受敬于海峡两岸，是极为重要的政治纽带。每年的十一等重大节庆日，孙中山的画像都出现在北京天安门广场上。
		我们希望蔡英文能够施加影响力，阻止民进党“立委”集体为高的提案开绿灯，针对她“维持现状”的诺言立木为信。毫无疑问她作为民进党主席有这个能力。如果她暗中支持这一提案在台“立法院”最终通过，那么她在今后两岸关系中将几无信用可言，两岸“台独”和反“台独”的新一轮斗争将打响起跑枪。
		若“废国父”提案本周末通过，大陆方面应当采取一个强有力的反制行动，给民进党以沉痛教训。虽然国民党仍在台上，“5·20”才做政权交接，但大陆反制带给海峡对岸的损失将被公众普遍记到民进党和蔡英文的头上。大家能看得懂。
		相信大陆这边已经积攒了足够多能够展现我们反“台独”意志的手段和工具。从现在开始，将有一段五花八门“去中国化”表演出笼的高危期，大陆方面应当严阵以待，随时对“台独”的冲动予以痛击，给民进党执政期内的两岸政策游戏立规矩。
		大陆用不着担心我方的反制会进一步刺激“台独”的嚣张，我们需要清楚，台湾的“台独”势力是感化不了的，他们只认得胡萝卜或大棒。没有斗争的坚决做支撑，我们同民进党内“温和派”讲道理的线索就很难成立。
		北京应当通过各种渠道让蔡英文明白，如果民进党刚一控制“立法院”就搞出“台独”倾向明显的大动作，那么她一定会被回敬清晰无误的下马威，她以两岸经济合作不倒退为基础的施政计划就需重写。民进党如果拨倒第一块多米诺骨牌，那么它要准备承受自己亲手放出的各种不确定性。
		希望蔡英文不是第二个陈水扁，事实上她也不具有成为第二个陈水扁的条件。那么她应劝诫民进党不要像8年前那样搞一些极端游戏，让大陆挺累，他们自己更累。
		""").foreach(println)*/
		var text="""中共中央总书记、国家主席、中央军委主席习近平，中共中央政治局常委李克强、张德江、俞正声、刘云山、王岐山、张高丽出席会议。
　　习近平在会上发表重要讲话，总结2015年经济工作，分析当前国内国际经济形势，部署2016年经济工作，重点是落实“十三五”规划建议要求，推进结构性改革，推动经济持续健康发展。李克强在讲话中阐述了明年宏观经济政策取向，具体部署了明年经济社会发展重点工作，并作总结讲话。
　　会议指出，今年以来，面对错综复杂的国际形势和艰巨繁重的国内改革发展稳定任务，我们按照协调推进“四个全面”战略布局的要求，贯彻落实去年中央经济工作会议决策部署，加强和改善党对经济工作的领导，坚持稳中求进工作总基调，牢牢把握经济社会发展主动权，主动适应经济发展新常态，妥善应对重大风险挑战，推动经济建设、政治建设、文化建设、社会建设、生态文明建设和党的建设取得重大进展。经济运行总体平稳，稳中有进，稳中有好，经济保持中高速增长，经济结构优化，改革开放向纵深迈进，民生持续改善，社会大局总体稳定。今年主要目标任务的完成，标志着“十二五”规划可以胜利收官，使我国站在更高的发展水平上。
　　会议认为，认识新常态、适应新常态、引领新常态，是当前和今后一个时期我国经济发展的大逻辑，这是我们综合分析世界经济长周期和我国发展阶段性特征及其相互作用作出的重大判断。必须统一思想、深化认识，必须克服困难、闯过关口，必须锐意改革、大胆创新，必须解放思想、实事求是、与时俱进，在理论上作出创新性概括，在政策上作出前瞻性安排，加大结构性改革力度，矫正要素配置扭曲，扩大有效供给，提高供给结构适应性和灵活性，提高全要素生产率。"""
		//text="""中共中央总书记"""
		text="""
		台湾民进党“立委”高志鹏近日提出3项修法提案，主张废除学校、公务单位悬挂国父遗像的规定，未来“总统”就职将不再向国父遗像行礼。这项提案预计26日将在“立法院”闯关。
		高志鹏是陈水扁嫡系人马，1993年起开始加入陈水扁“立委”办公室，2002年起担任台湾“立委”。他在上届“立法院”曾提案修改“国徽国旗法”“宣誓条例”等等，但遭国民党团阻挡，无法进入审查程序。此次他卷土重来，由于民进党已在“立法院”掌握多数席位，如果民进党集体支持，该提案就将通过。
		此事在台湾激起轩然大波，蓝营人士及媒体痛斥高的提案，要求蔡英文表态，蔡于昨天做出态度模糊的回应，不置可否。而《自由时报》等绿营力量则嘲笑孙中山是“只挂在墙壁上的国父”，并且宣扬孙中山几次来台都与台湾的命运实际无关。台《联合报》质疑民进党改走“装饰主义台独”，认为民进党如今不敢真枪实弹迈向“建国之路”，只好忸忸怩怩改走曲径幽巷，因而选举之后“转型正义”如雨后春笋冒出头。
		今年11月12日是孙中山先生诞辰150周年，大陆方面已宣布将举行隆重纪念活动，而台湾民进党却有可能将胜选后的第一刀砍向这位中国民主革命的伟大先驱。
		蔡英文胜选前后多次宣称将致力于“维持现状”，然而一旦孙中山的国父地位从台湾正式场合消失，就意味着释放出将用切香肠方式把她所承诺维护的现状一步步搞没的信号。
		尊崇国父是“中华民国”延续了70几年的传统，它如今是台湾公开标志自己“中国血统”为数不多的政治符号之一。孙中山还长期受敬于海峡两岸，是极为重要的政治纽带。每年的十一等重大节庆日，孙中山的画像都出现在北京天安门广场上。
		我们希望蔡英文能够施加影响力，阻止民进党“立委”集体为高的提案开绿灯，针对她“维持现状”的诺言立木为信。毫无疑问她作为民进党主席有这个能力。如果她暗中支持这一提案在台“立法院”最终通过，那么她在今后两岸关系中将几无信用可言，两岸“台独”和反“台独”的新一轮斗争将打响起跑枪。
		若“废国父”提案本周末通过，大陆方面应当采取一个强有力的反制行动，给民进党以沉痛教训。虽然国民党仍在台上，“5·20”才做政权交接，但大陆反制带给海峡对岸的损失将被公众普遍记到民进党和蔡英文的头上。大家能看得懂。
		相信大陆这边已经积攒了足够多能够展现我们反“台独”意志的手段和工具。从现在开始，将有一段五花八门“去中国化”表演出笼的高危期，大陆方面应当严阵以待，随时对“台独”的冲动予以痛击，给民进党执政期内的两岸政策游戏立规矩。
		大陆用不着担心我方的反制会进一步刺激“台独”的嚣张，我们需要清楚，台湾的“台独”势力是感化不了的，他们只认得胡萝卜或大棒。没有斗争的坚决做支撑，我们同民进党内“温和派”讲道理的线索就很难成立。
		北京应当通过各种渠道让蔡英文明白，如果民进党刚一控制“立法院”就搞出“台独”倾向明显的大动作，那么她一定会被回敬清晰无误的下马威，她以两岸经济合作不倒退为基础的施政计划就需重写。民进党如果拨倒第一块多米诺骨牌，那么它要准备承受自己亲手放出的各种不确定性。
		希望蔡英文不是第二个陈水扁，事实上她也不具有成为第二个陈水扁的条件。那么她应劝诫民进党不要像8年前那样搞一些极端游戏，让大陆挺累，他们自己更累。
		"""
		text="""
        王国庆：

　　2016年，全国政协将进一步聚焦经济发展，宏观经济形势分析座谈会今年将每季度举办一次，比去年增加两次。委员们还将围绕着“降低实体经济企业成本、推动供给侧结构性改革”、推动“大众创业、万众创新”、“优化金融服务、支持创业创新”等专题开展深入的调研并提出意见、建议。你问的第三问，关于供给侧改革的问题，这是一个很专的问题，也是很大的课题。3月6日下午三点，我们在这里专门安排一场记者会，有这方面的委员专家和大家深入讨论这一话题，你一定参加。

　　2016-03-02 15:22:44

　　香港大公报、大公网记者：

　　国家层面的“十三五”规划马上公布，香港很多市民眼见内地城市经济社会快速发展心里感到很着急，请问发言人“十三五”期间中央政府将采取哪些具体措施支持内地和香港的合作？谢谢。

　　2016-03-02 15:23:55

　　王国庆：

　　“十三五”规划主要是内地经济社会发展规划，但是一定会考虑港澳因素及其发展需要。中共中央已经提出“十三五”规划的建议，里面明确指出要发挥港澳独特优势，提升港澳在国家经济发展和对外开放中的地位和功能。这凸显了中央政府在谋划国家整体发展布局时对港澳的高度重视和热切期待。今后，中央政府将一如既往地支持港澳地区发挥优势，参与国家建设。港澳特区政府和社会各界也要齐心协力，巩固原有优势、开发新优势，注意做好与国家战略的对接，找到“国家所需、港澳所长”的交汇点。谢谢。

　　2016-03-02 15:24:31

　　中央电视台、央视网、央视新闻客户端记者：

　　刚刚过去的地方“两会”我们注意到这样一个现象，很多政协委员履职过程中呈现出“一冷一热”，有些委员几年不参会也没有提案，但有些委员为了争取发言机会有的写口号，有的打标语。在冷热之间，我们注意到在3天前规范政协委员履职的一个规定已经在常委会通过，是不是意味着全国政协对委员履职真正的含金量会有一个科学评价？什么样的委员不及格，发言人能不能描述一下。

　　2016-03-02 15:25:02

　　王国庆：

　　谢谢媒体朋友关注委员履职，包括对我本人的监督。全国政协历来重视自身队伍建设，包括委员履职能力建设，在加强和改进学习培训的同时，也注重制度建设，推进委员履职的制度化、规范化、程序化。去年全国政协制定和修订了委员履职、专委会工作、提案办理协商、委员视察、专题调研和反映社情民意等十项制度。你提到刚刚闭幕的政协十二届常委会第十四次会议已经审议通过了《全国委员会委员履职工作规则(试行)》，规则中对委员履职的内容、履职的方式、履职保障、履职管理都作出了规定。这个文件“两会”后就会公布，我想等这个文件公布的时候，可以请我们新闻局再安排一次请有关负责同志解读一下文件，这个问题可以说的更明确。全国政协强调政协委员应该懂政协、会协商、善议政，而且更应该守纪律、讲规矩、重品行，必须时时以高度的政治责任感、强烈的委员意识和良好的精神风貌认真履职、扎实工作。在这一点上，我们诚恳地欢迎记者朋友和广大人民群众监督。

　　2016-03-02 15:25:26

　　中国新闻社和中新网记者：

　　中国的医药卫生改革已经进行了一段时间了。作为普通百姓还是不敢生病，好像获得感没有那么强烈。作为全国政协在推动医药卫生改革方面有哪些想法？

　　2016-03-02 15:25:56

　　王国庆：

　　你提了一个公众很关心的问题，我们梳理这次新闻发布会问题的时候发现这个问题是大家很关注的问题。我国的新医改从2009年启动以来，从中央到地方下了很大功夫，也做了不少探索，也应该说有了很多实实在在的进展。前不久召开的中央全面深化改革领导小组第21次会议强调，要把是否促进经济社会发展，是否给人民群众带来实实在在的获得感作为改革成效的评价标准。我认为，按照这个评价标准，我国的医改确实还有很长的路要走。我们必须看到，医改是一项长期、艰巨、复杂的系统工程，在世界各国都是个大难题，更何况在一个有13亿人口的发展中大国。近年来，全国政协始终高度关注医改的推进，我们每年都有医改相关的调研考察、双周协商座谈，今年全国政协已经将“深化医药卫生体制改革”列为重点协商议题。去年年底的第38次主席会议上已经将就做好这次专题调研，开好这次专题协商会做出了具体部署。今年将有3位副主席分别带队到全国各地，深入到医药卫生一线和广大公众当中进行深入调研。要摸实情、听呼声，为我国的医改深入推进谋实策、出实招。大家可能注意到，我们有一路人马由韩启德副主席带队已经在福建、安徽进行调研，这个调研还在继续。这个话题是个大话题，里面要讨论的内容很多。我想我们调研完了以后对这个问题可能有更多更有说服力的说法。

　　2016-03-02 15:27:52

　　团结报、团结网记者：

　　美国等西方国家不断指责中国在南海地区实施军事化，指责中国的行为影响南海地区的航行自由，甚至有美国军方人士指责中国在东亚谋求霸权，请问发言人您怎么看待这一问题？谢谢。

　　2016-03-02 15:28:19

　　王国庆：

　　南海问题外交部和国防部的发言人已经多次表态，全国政协完全支持他们所表述的立场。全国政协委员当中有不少常年在涉外领域工作的同志，也有很多这方面的专家学者。对这个问题我们是有共识的。南海应该成为和平之海、友谊之海、合作之海，南海问题不应该成为个别国家用来遏制中国发展的借口和工具，这一点是很明确的。谢谢。

　　2016-03-02 15:29:26

　　塔斯社记者：

　　有两个问题。第一，我们比较关注中国经济发展，有人认为现在外国的经营包括俄罗斯人在中国的经营环境比以前差，甚至有外企认为在中国遇到不公平对待，您怎么看待这个问题？第二，西方专家认为中国经济面临“硬着陆”，你怎么理解“硬着陆”，中国是否面临“硬着陆”？

　　2016-03-02 15:31:06

　　王国庆：

　　第一个问题，中国领导人最近在好几个国际场合已经反复重申，中国利用外资的政策不会变，为各国在华投资企业创造良好投资环境的政策不会变，保护外商投资企业合法权益的政策不会变。年底前根据商务部的统计，2015年中国吸引外资同比增长5.6%，尽管放缓但还是增长。去年1月份到11月份，在中国设立的外商投资企业同比增长11%。从资本驱利性这一点来判断，这组数据是不支持“外企在华营商环境趋于恶化”的说法。全国政协这两年也组织委员对外企在华的营商环境做实际调查，我本人就跟政协外委会调研组到过上海、江苏，在跟外企直接的接触当中，我的印象，包括我们去调研的委员的印象，外企在华营商的环境并没有恶化，而且在不断地优化。只是随着社会主义市场经济体制的不断完善，人民生活水平提高后需求的变化，再加上一些要素成本的上升，市场竞争变得更加激烈了，钱有点不那么好赚了，这是事实。但是我想说中国政府为来华投资创业的外商，打造法治化、国际化、便利化的营商环境，这个坚定信心不会变。相信中国这个有13亿人口的大市场，未来仍将继续成为全球受青睐的投资目的地，在这儿还是可以赚钱的，大家可以放心在这儿投资、创业。

　　第二个问题，我也看到境外有一些舆论炒作中国“硬着陆”，这个概念不存在。从我前面讲到的第二个问题中已经讲到中国经济总体情况是好的，所以不存在“硬着陆”的问题，他们担心多了，如果别有用心想“唱衰”中国，可能结果不会让他满意。中国的经济是好的，现在有困难，但是我们会保持中高速的增长。

　　2016-03-02 15:35:51

　　中国青年报、中青在线记者：

　　近年来，我们留意到时常可以听到关于公务员、法官、国企、金融机构的高管以及媒体人士离职的消息，请问发言人您怎么看待这种“离职潮”的现象？

　　2016-03-02 15:36:19

　　王国庆：

　　你刚才用了一个“离职潮”，我注意到有些媒体报道个别现象的时候喜欢用“离职潮”这三个字，我不太赞同。我觉得没有成“潮”，只是一种辞职现象，我想应该从两方面看，一方面这反映了新时期人才流动的一个新特点，我觉得我们应该为人才流动喝彩，这是好事。中共十八届三中全会提出要打破体制壁垒、扫除身份障碍，让人人都有成长成才、脱颖而出的通道，让各类人才都有施展才华的更广阔天地。一个真正的世界强国一定是人才自由流动的国家。这几年来社会鼓励创业的气氛越来越浓，创业创新蔚然成风，这有利于人力资源的优化配置，社会的健康发展。当然从另一方面看，各类用人单位要真正树立重视人才的观念，国以才立，政以才治、业以才兴。改革中要注重创造有利于人才成长和施展才华的环境，为事业健康长远发展要留住人才、用好人才。因为社会人才总是有限的，所以其实一方面要鼓励流动，另一方面要想办法留住，这两边都要把握好，这应该是一个健康的人才观。谢谢。

　　2016-03-02 15:41:12

　　紫光阁杂志社记者：

　　中国启动不文明游客黑名单制度以来，已经有十多人榜上有名，你曾在全国政协大会上作过发言，提出中国人是中国故事最好的讲解者，请问怎样让每个人都能讲好中国故事呢？

　　2016-03-02 15:41:26

　　王国庆：

　　谢谢你的问题，我想起在网上看到这样一个故事。美国哥伦比亚大学东亚系华工，在当今美国的汉学研究领域里面是赫赫有名的，但是很少人知道东亚系的前身其实是由一名华人劳工建议并且捐款建立的，这是美国大学里面的第一个中文系，当然他的捐款肯定不够了。据记载这是110多年前的事儿，这名华工的英文名字是“Dean Lung”(丁龙)。为什么他要提议呢？他说为了让美国人更好地了解中国和中国文化。就是为了这一点他把差不多一生的积蓄——12000美元都捐给了哥伦比亚大学，要求在该校设立汉学教育，这个故事令我很感动。时至今日，让世界更好地了解中国，特别是发展、变化中的中国，仍然是一项十分艰巨的任务。怎么讲好中国故事？首先要有充分的自信，要把中国的发展变化理直气壮地讲好。因为历史和实践都已经证明我们选择的这条道路是正确的。其次是要转变话语体系，要用国际化的表述把中国故事讲好，讲的让人听得懂、听得进，听了还得信。

　　我们每个人都应有讲好中国故事的责任意识，因为你的一言一行、一举一动其实都是在演绎中国故事。我很赞同这么一个说法，中国人就像一本厚厚的书，每个中国人都是这本书当中的生动一页。我以为每个中国人都应该不断提高讲中国故事的责任意识、主动意识、能力水平，自觉地把中国故事讲好。

　　2016-03-02 15:42:35

　　凤凰卫视和凤凰网记者：

　　大家都知道春节期间香港发生了旺角暴乱，这个事件在内地和香港都引起了极大震惊。现在过去一段时间以后，大家在震惊当中反思背后是否有深层次的原因，不知道中央政府是怎样评估的？全国政协有什么看法？是否涉及到中央政府对香港的政策，有一些政策是否会发生变化？全国政协有很多港澳委员，港澳委员对他们自己所处地区的社会稳定和经济发展，如何肩负一些更加重要的责任，您认为他们应该怎么做？谢谢。

　　2016-03-02 15:45:26

　　王国庆：

　　香港是个开放的多元社会，不同界别、阶层的人士对经济、政治、社会等领域有不同看法，这都是正常的。但谋发展、保稳定、促和谐是香港广大市民的共同愿望，也是香港的根本利益所在。我们反对少数人采取违法甚至暴力手段妄图搞乱香港，破坏内地和香港的合作和交流。香港回归将近19年了，19年来“一国两制”取得的成就是举世公认的。中央贯彻“一国两制”的方针始终坚持两点，一是坚定不移、不会变、不动摇；二是全面准确，确保“一国两制”在香港的实践不走样、不变形，始终沿着正确的方向前进。总的来说，维护国家的主权、安全和发展利益，保持香港长期繁荣稳定是“一国两制”方针的根本宗旨，也是实现中华民族伟大复兴中国梦的重要组成部分。

　　刚才提到港澳委员在其中如何发挥作用。港澳地区的全国政协委员是港澳社会各领域的精英，也是爱国爱港、爱国爱澳的优秀代表，在港澳实践“一国两制”的伟大事业当中担负着重要角色和重大责任。在这方面，张德江委员长、俞正声主席都提出过明确的指导性意见。概括起来有几个方面：一是坚持全面准确理解和贯彻“一国两制”方针，带头维护宪法和基本法权威；二是坚守爱国爱港、爱国爱澳立场，坚决维护国家主权、安全利益，维护特区长期繁荣稳定；三是全力支持行政长官和特区政府依法施政，促进爱国力量的团结和发展壮大，引导港澳社会各界抓住机遇发展经济、改善民生、促进和谐；四是不断推动港澳与内地的交流合作与民众的团结和谐，为内地和港澳共同发展贡献力量。

　　2016-03-02 15:51:06

　　尼日利亚记者：

　　今天上午在新闻中看到有一些中国公司遇到了裁员的问题，另一方面在新闻中看到，中国现在出台了全面二孩政策，这就意味着中国人口会越来越多，对失业和人口越来越多这两方面的问题，全国政协有没有什么具体的建议处理好二者间的关系？

　　2016-03-02 15:57:43

　　王国庆：

　　谢谢你的提问，你关心的这个问题是个长远的问题。二孩政策放开以后，人口会有适当的增长，这是不争的事实。但是从长远来讲，也是国家经过判断以后才作出的这样一个决策。至于你说现在有一些企业、有一些公司员工下岗，我想这是暂时的。政府会协调好这方面的关系，保持人口的有序增长，保证经济的持续发展。这个问题如果要讲可以讲很多话，可以找很多数据，我建议你下次有机会采访政协委员当中的专家，他们会给你更权威的回答。

　　2016-03-02 15:58:24

　　中国网记者：

　　大家注意到在“一带一路”战略部署和亚投行建成之后，国家又有一张新牌是国际产能合作，国外媒体对它有不同的解释，请问您怎么看这个问题？

　　2016-03-02 15:58:37

　　王国庆：

　　谢谢你的提问，关于国际产能合作的意义、总体要求和主要任务，国务院去年5月份已经出台过一个文件《关于推进国际产能和装备制造合作的指导意见》，文件中已经作了明确阐述。政协委员怎么看？去年全国政协组织委员就这一专题在国内、国外作过多项调研。许多鲜活的案例都说明，面对世界经济艰难复苏的严峻形势，各国必须同舟共济，要不断扩大利益汇合点。中国有很多质优价廉的装备和产能“走出去”，这是好事，一方面有利于我国顶住经济下行的压力，扩展更大的发展空间。同时产能“走出去”也有利于相关国家加快发展、扩大就业。

　　有一个很好的例子，去年9月全国政协领导的中国经社理事会参与了在印尼首都雅加达举办的“中国—东盟产能高层论坛”。会上发言的代表都认为中国和东盟双方在产能合作方面基础好、潜力大，有利于推动东盟各国经济结构转型和产业升级，促进地区经济融合和发展。目前，中国同哈萨克斯坦、埃及等十多个国家签署了国际产能合作协议，中国和哈萨克斯坦两国的产能合作已经达成52个项目，总金额超过240亿美元。谢谢。

　　2016-03-02 16:06:09

　　人民日报、人民网和人民日报客户端记者：

　　去年年底中央召开扶贫开发工作会议提出精准扶贫、精准脱贫，我们注意到一些地区尤其是少数民族地区他们的经济社会发展严重滞后于其他地区，对于这些地区的脱贫全国政协能够发挥哪些作用？

　　2016-03-02 16:06:39

　　王国庆：

　　确实有些民族地区的经济社会发展相对滞后，是实现全面建成小康社会“短板”当中的“短板”。习近平总书记曾经强调，全面实现小康，决不能让一个少数民族、一个地区掉队。全国政协对加快民族地区的发展高度重视，委员中有这样一句话：对民族贫困地区要“高看一眼、要厚爱一分”。

　　2016-03-02 16:07:05

　　王国庆：

　　说到这儿我又要讲故事。不知道是否各位听说过“包虫病”？这是一种在我国部分高寒、干旱、少雨农牧区比较常见的人畜共患的寄生虫病，因为患病感染率、致死率都比较高，所以被称作为“虫癌”。这种病也成为制约地区经济发展的重要公共卫生问题。有一些家庭因为有人患了这种病而致贫、返贫。去年全国政协就包虫病防治组织委员到青海、四川、新疆、宁夏四个省区进行调研。委员们深入到牧区，和牧民同吃同饮，到家里、到医院看望病人，当地的干部群众都十分感动。只有这种深入的调研，所以他们能摸到实实在在的情况。调研回来以后提出的意见建议为有关部门高度重视，国家卫计委、科技部已将包虫病防治列入国家重大计划，并启动包虫病综合防治试点工作。这说明，人民政协在推动精准扶贫、精准脱贫方面大有可为。今年全国政协将组织好“实施精准扶贫、精准脱贫，提高扶贫实效”专题议政性常委会，重点就推进精准扶贫决策的贯彻落实、武陵山片区精准扶贫、加强草原生态系统保护和修复、民族地区绿色发展等问题开展调研，为破解“瓶颈”，补齐短板，献计出力。谢谢。

　　2016-03-02 16:13:23

　　中国日报社记者：

　　官方数据显示2015年空气质量好于往年，但很多人发现生活当中可能没有很明显的感受到空气质量的显著改善，尤其是北京，连续两次发布了空气重度污染的红色预警。请问发言人雾霾治理何时才能取得实质性的成效？谢谢。

　　2016-03-02 16:15:34

　　王国庆：

　　确实，我们在座各位对雾霾天气都非常敏感。刚才你提到有统计数据，但是没有感受到空气质量有明显的改善。我想说统计数据确实告诉我们总体情况是在向好的方向变化。以北京为例，2015年北京空气质量达标天数比2014年增加了14天，重污染天数较上一年减少了1天。但是为什么大家感觉统计数据和我们的感受有反差？为这个问题我请教了权威人士，他给我算了一下账我才明白，他告诉我去年前10个月情况不错，达标天数同上一年同期相比增加了31天。可是后两个月情况就糟糕了，后两个月同比减少了17天。一增一减全年就14天了。在年底北京两次启动红色预警，人们的感受可想而知了。在这儿我们还要非常客观的说，中国在治理雾霾方面已经做了很多，但是需要做的更多。
        """
		text="""毛泽东、朱德以习近平同志为总书记的党中央紧紧围绕坚持和发展中国特色社会主义这个主题，厉以宁、何来李克强丙申金秋硕果累累。王安全，张雨绮潇洒女王吴亦凡。
  瞿颖颖、王祖贤！孙楠楠，刚才王国庆提到有统计数据，王五来了，赵王发髻。程增谦，王宇鑫李姗姗和张馨。杨阿姨。国庆说蔡国庆去了。孙开伟精通hadoop的组成及底层原理及优化。赵贤鹤熟悉常见业务处理，
  能快速融入团队。杜小姐多年c++前后台陈留王开发经验。历朝封陈留王者大概26人，其中著名者有刘协、曹奂等。克强指出陈阿姨。舟山市充分发挥本地深水港口资源优势，与香港华光航运集团公司合作，于去年底在马峙锚地建立了外轮清舱基地。
  中国科学院学部委员、复旦大学副校长、著名数学家谷超豪教授，近年来从事规范场数学结构的研究，取得深入、系统的、处于国际前沿的研究成果。
津门另四名选手滕新宇、马槟、刘书志、刘振刚也将于明日角逐各路强手。与解放军队交锋，上场前，解放军队领队一看天津队的高大中锋王兴华未换比赛服，喜形于色地向场上队长庄连胜说："王兴华不上场，可以让小贾（贾秀全）多出击。"
还要开设老舍先生名著《茶馆》中描述的"裕泰茶馆"，以北京人艺演出的话剧《茶馆》的舞台布景设计为蓝本，并置２７个按北京人艺演员扮相塑造的茶馆人物蜡像于其中。
岭南画派创始于辛亥革命后的高剑父、高奇峰兄弟及陈树人、何香凝等广东画家，是近百年来一个重要画派。陈有炳１９４７年生于新加坡，早年曾师事于名画家范昌乾，１９７３年任华翰研究会主席，现在新加坡南洋美术专科学院任教。
云南省楚雄彝族民间艺术团，于近日赴日本冲绳县参加"第三届亚洲艺术节--石垣９０"。大家在发言中一致热烈拥护华主席为首的党中央关于发展我国社会主义科学技术事业的英明战略决策，热烈祝贺这次全国科学大会的园满成功。
胡鞍钢和王毅强调指出，这种模式绝不是一种理想的模式，也并非是出于自愿的选择，它是在中国既定的人口、资源总量和发展水平的限制下迫不得已的选择，也是唯一的选择，它与消耗、破坏我们的子孙后代赖以生存和发展的自然支持系统和资源基础的巨大代价相比，为中华民族的千秋万代长远计议，是值得的和必要的。
思念好友的思绪使我在招待所的床上辗转反侧，终于爬起来给小辛写了一封信，叮嘱完毕，才安然入睡。参加会见的还有中央有关部门、解放军总部、武警部队总部负责同志赵东宛、于永波、刘安元等。
国家南极考察委员会办公室主任郭琨，宣读了国家南极考察委员会关于给秦大河同志记一等功的决定。美国总统布什和英国首相梅杰经过两天的会谈后于今天表示，两国将共同努力，要求联合国通过决议，授权在波斯尼亚上空实行军事禁飞，并防止科索沃和马其顿发生流血冲突。
太湖的治理，当前要继续抓好望虞河、太浦河、杭嘉湖南排和环湖大堤等四项主要工程的建设，望虞河、太浦河按设计于１９９４以前开通。
舟山市充分发挥本地深水港口资源优势，与香港华光航运集团公司合作，于去年底在马峙锚地建立了外轮清舱基地。若按我们传统眼光来看，被中国球迷亲切地称为"施大爷"的这位主教练的行为，至少宜于扣上这么几顶"帽子"。
会议还听取了政协全国委员会秘书长彭友今关于对各级政协委员的政策落实情况进行检查的意见，通过了有关批准政协六届全国委员会各委员会、工作组机构及人事安排等事项。
波黑冲突交战三方定于１５日下午在欧共体调解人卡林顿的主持下举行和谈。他说，邓大姐在中纪委工作期间，高度重视党风党纪建设，孜孜不倦地致力于维护党规党法的艰巨工作，提出要加强党的纪律，坚定不移地搞好党风，反对和纠正以权谋私、严重官僚主义等党内不正之风，为拨乱反正、恢复和发扬党的优良传统和作风、严明党的纪律，做出了巨大的贡献。
中国科学院学部委员、复旦大学副校长、著名数学家谷超豪教授，近年来从事规范场数学结构的研究，取得深入、系统的、处于国际前沿的研究成果。
沈明观请人为他通读了《怎样养好长毛兔》一书；拜访有经验的养兔专业户，克服了喂兔、治兔病、剪兔毛、母兔配种、护理小兔、修理兔棚等工作中的各种困难，终于成了盲人养兔能手，兔场逐年盈利。
深入研究道教斋醮的仪式，或许能有助于加深对从秦到唐的文化史的了解。当然真正的乐趣还在于尝一尝自种的西红柿、黄瓜、白菜和胡罗卜之类的滋味，司徒雷登闻一闻亲手栽培的各种鲜花的幽香。
来函照登今年五月三十日，本报关于《吴增谦等人谋求自己出访未成，阻挠高级工程师商善最出国参加会议》的报道发表后，司马相如说引起社会各方面的较为强烈的反响。
政协全国委员会常务委员会关于进一步认真学习《邓小平文选》的通知。他说，邓大姐在中纪委工作期间，高度重视党风党纪建设，孜孜不倦地致力于维护党规党法的艰巨工作。
世界各国通讯社和报纸广泛报道了中共中央关于"七五"计划的建议和赵紫阳关于这个建议所作的说明，认为这个建议表明中国将继续大力推行经济改革和对外开放的政策。
慕尼黑是名副其实的"啤酒城"，这里不仅酒店林立，而且拥有最古老的啤酒店─-建于一五八九年的霍夫布芳恩馆，拥有世界最大的啤酒店─-马台泽，共有五千个座位。
奥地利于十二世纪形成公国，一八六六年在普奥战争中战败后，建立奥匈帝国。革命导师列宁，曾于１９１９年１２月写过一篇《关于星期六义务劳动》的著名论文。
昨天下午５时，刚从松江县受灾现场回来的上海气象局副局长、高级工程师唐新章打电话告诉报社，由于习俗的偏见，离了婚的女人往往会把重新结婚"看成是一种灾难"；"在某些范围内，我们到现在还没有击败孔老二"。"
爱泼斯坦说："龚澎善于利用一切机会来做工作，她往往实际上是在给舆论以正确的引导，但并不使人感到你是在被人引导。"
南齐诗人谢朓这首《咏蒲》诗，极其精炼地描摹出菖蒲的主要特征。王玉林; 叶迪生,王玉霞,张士珍,于淑珍,郝培嵩; 在会上发言的有天津市特等劳动模范。这厂发往武钢的６１０冷轧辊合格率达百分之九十七点六，
中国民航局局长沈图率领民航工作组一行和机组人员共三十三人，深山区的观音堂公社李台大队把林果收入中的按劳分配部分拿出来，石家庄地区农村开展了一场关于专业户代表米秋喜应不应担任党支部书记的讨论。
联合体成批涌现，米秋喜式的人物成千上万，听取各代表团审议张曙光省长所作的《政府工作报告》、省计划委员会主任张震环作的《关于一九八四年国民经济和社会发展计划草案的报告》、
省财政厅厅长周国卿作的《关于一九八三年财政决算和一九八四年财政预算草案的报告》的意见；讨论关于代表要求召开幼儿教育座谈会和卫生战线代表座谈会的问题；决定在大会上作提案审查报告。
这期间，袁教授先后八次来厂作指导，终于制成大型壁画《山魂水魄》，被评为全国优秀壁画。廊坊市葛鱼城镇于堤村五十一岁的农民、共产党员胡连华，靠种植树苗走出一条致富路。
白涧区的领导同志告诉我们，白涧是革命老区，山区的最终出路在于可再生的林牧，林牧发展不起来，晋城的块煤、邢台市的球磨铸造生铁、濮阳的石油和天然气、菏泽地区的山羊皮等资源优势。
支部书记李景保、村主任陈志太和其他支委都积极地开展了自我批评，去年十一月，小郝得知这件事后，主动与有关部门联系，李木同平时主持正义，将四川省西南地区生产技术处的郝长年从卧铺车的上铺摔了下来。
苏联著名游船－"高尔基"号轮，于最近翩然驶入秦皇岛港。低于居里温度的结论，紫的是聚峰……老赵介绍说，发行了一套慈禧生辰纪念邮票。这套邮票的设计者是供职于海关的德国人费拉尔，
谁知管仲一死，齐国被竖刁、易牙、开方等奸臣掌权，国内大乱，"威公（即齐桓公）薨于死，……齐无宁岁"。衷心祝愿全党老干部，象"肖何举曹参以自代"那样，充分利用宝贵的晚年，
满腔热情地把主要精力用于对中青年干部的传帮带，为解决好干部队伍的交接班问题，为我们党的事业后继有人、兴旺发达作出更大的贡献。大队长高忠义因公骨盆摔伤，马胜利在改革的实践中，
为社会主义现代化建设做出了宝贵的贡献，勾践所做的长期准备，就是在努力改变吴越双方条件和力量对比，最后由吴强越弱，转化为吴弱越强，终于灭了吴。
古希腊哲学家阿那克西曼德的朴素进化论思想被扼杀，和１８世纪法国进化论思想先驱布丰放弃所有与神学观点相冲突的说法，楚人把白天测量时的正确认识应用于深夜，
但是，德谟克利特原子论中也有形而上学的倾向，因此，亚里士多德否定了柏拉图的理念论，黑格尔在谈到辩证法的具体同一和形而上学抽象同一性的区别时说得好：
正常的新陈代谢的生理机能被破坏，如此层层剥皮，林农得不到实惠，金源国际谷物有限公司』成立签字仪式，于十一月三十日在京举行。
该系列产品具有清洁、滋润、防水、防霉特点，他们生产的文君酒系列产品曾经在法国、香港等地举行的国际博览会上获奖。可是，原告方郎酒厂原厂长、法人代表云宗倜、办案人林永清则说，
横的方面：这里不是指由大田承包发展到林牧副渔各业的承包，南部、东南部路间的方田，每５０多米便有一条笔直的长长渠道，一米多深，三米多宽，渠岸高垒着新翻的泥土；北部一些土地上，
散落着一个个白色筒状物，一问，方知是地下输水管道的给水栓；方田灌溉渠道同排涝渠一样笔直，只是窄小些且略高于田面；浇灌过的麦田苍绿中泛着薄冰的白光，大田则黄中透黑。
处于闽江入海口的乌龙江江心的郊区建新乡冠洲村受洪水围困，８２４名群众束手待援。鄄城县一位孤身汉族老汉于大海，曾到西马垓帮过一段时间的工。位于江汉平原的总后某基地，
报纸发表大会消息的当天晚上，市委第一书记张建尧同志亲自召集纺织系统的各级干部，一道学习华主席关于“我们一定要高举毛主席树立的大庆红旗”的题词和李先念同志在全国工业学大庆会议上的开幕词，研究怎样大干快上。
高举毛主席树立的大庆红旗，把石油工业推向一个新的大发展时期。鉴于陆东明同志道德败坏，抓紧铁路、公路和港口等设施的改建、扩建工程，
今年又先后在福建、广西、湖南、安徽、甘肃、新疆、河南、吉林和辽宁正式建立了记者站，赵紫阳会见西班牙客人时指出中共中央代总书记、国务院总理赵紫阳指出，
无私无畏勇顶妖风战恶浪陈毅同志在长期革命斗争实践中同形形色色的敌人较量过，由于周总理的坚决保护和无微不至的关怀，林彪、“四人帮”一伙妄图对陈毅同志下毒手的阴谋才未能得逞。
徒步５０００多公里考察雅鲁藏布江，携着各种修理器具来到位于西长安街的茂林小区。哈队２号宫明终于远射命中，打破僵局。历史学家周谷城教授也愤怒地指出：这是姚文元陷人于罪。
""".replaceAll("\\s","")
  text="""中共中央总书记、国家主席、中央军委主席习近平，中共中央政治局常委李克强、张德江、俞正声、刘云山、王岐山、张高丽出席会议。
　　习近平在会上发表重要讲话，总结2015年经济工作，分析当前国内国际经济形势，部署2016年经济工作，重点是落实“十三五”规划建议要求，推进结构性改革，推动经济持续健康发展。李克强在讲话中阐述了明年宏观经济政策取向，具体部署了明年经济社会发展重点工作，并作总结讲话。
　　会议指出，今年以来，面对错综复杂的国际形势和艰巨繁重的国内改革发展稳定任务，我们按照协调推进“四个全面”战略布局的要求，贯彻落实去年中央经济工作会议决策部署，加强和改善党对经济工作的领导，坚持稳中求进工作总基调，牢牢把握经济社会发展主动权，主动适应经济发展新常态，妥善应对重大风险挑战，推动经济建设、政治建设、文化建设、社会建设、生态文明建设和党的建设取得重大进展。经济运行总体平稳，稳中有进，稳中有好，经济保持中高速增长，经济结构优化，改革开放向纵深迈进，民生持续改善，社会大局总体稳定。今年主要目标任务的完成，标志着“十二五”规划可以胜利收官，使我国站在更高的发展水平上。
　　会议认为，认识新常态、适应新常态、引领新常态，是当前和今后一个时期我国经济发展的大逻辑，这是我们综合分析世界经济长周期和我国发展阶段性特征及其相互作用作出的重大判断。必须统一思想、深化认识，必须克服困难、闯过关口，必须锐意改革、大胆创新，必须解放思想、实事求是、与时俱进，在理论上作出创新性概括，在政策上作出前瞻性安排，加大结构性改革力度，矫正要素配置扭曲，扩大有效供给，提高供给结构适应性和灵活性，提高全要素生产率。过去，母亲齐心竭尽全力营造一个温馨的家庭环境，使得他父亲习仲勋能够集中精力工作。"""
	text="""
2016年12月12日，第一届全国文明家庭表彰大会在京举行。中共中央总书记、国家主席、中央军委主席习近平亲切会见全国文明家庭代表，并发表重要讲话。习近平强调，我们要重视家庭文明建设，努力使千千万万个家庭成为国家发展、民族进步、社会和谐的重要基点，成为人们梦想启航的地方。人们常说“家和万事兴”“治国先齐家”。跟大家一样，习近平有着浓浓的“家国情怀”。他注重家庭、注重家教、注重家风，为大家做出了表率。无论时代如何变化，无论经济社会如何发展，对一个社会来说，家庭的生活依托都不可替代，家庭的社会功能都不可替代，家庭的文明作用都不可替代。”12月12日，习近平在会见第一届全国文明家庭代表时说。这并非习近平首次如此强调家庭建设。在2015年春节团拜会上的讲话中，习近平提出要“注重家庭、注重家教、注重家风”，就曾引发媒体高度关注，引起海内外人士的共鸣。一位泰国学者称，习近平在时代变革的背景下提出重视家庭建设，具有战略眼光、符合时代要求与社会现实。中华民族自古以来就重视家庭、重视亲情。中华民族传统家庭美德，铭记在中国人的心灵中，融入中国人的血脉中，习近平更是如此。他在一个和睦的家庭里长大。如今，他自己早也是一位父亲，愈加珍视家庭的幸福。在他的办公室里，几张不同年代的温馨家庭生活照，被放置在醒目的位置：他用轮椅推着年事已高的父亲，他牵着母亲的手在散步，他同夫人彭丽媛合影，他骑自行车载着年幼的女儿玩耍……他很孝敬父母。家人为父亲举办88岁寿宴时，当时习近平作为一省之长，公务繁忙，实在难以脱身，于是抱愧给父亲写了一封深情款款的拜寿信。母亲齐心如今也年过90岁高龄，习近平每当有时间陪她一起吃饭后，都会拉着母亲的手散步，陪她聊聊天。他很关爱妻子。妻子彭丽媛作为军旅歌唱家，那时经常要接受任务奔赴外地慰问演出，习近平总是十分牵挂，只要条件允许，无论多晚，他每天都要跟妻子至少通一次电话。过去每逢除夕，彭丽媛总要参加春晚演出，在外地工作的习近平只要回北京过年，就总是边看节目边包饺子，等她演出结束回家后才煮饺子一起吃。他知道，家庭和睦，益于事业。过去，母亲齐心竭尽全力营造一个温馨的家庭环境，使得他父亲习仲勋能够集中精力工作。现在，妻子彭丽媛对习近平也非常关心体贴。早年夫妇俩聚少离多，一有机会团聚，彭丽媛就想法子变花样给他做可口的饭菜。孝敬父母、爱护妻儿，习近平对家庭幸福看得如此之重。在他看来，这不是只关系一家一户的普通小事。“家庭和睦则社会安定，家庭幸福则社会祥和，家庭文明则社会文明。我们要认识到，千家万户都好，国家才能好，民族才能好。”1969年，刚满15岁的习近平，插队到了延安。在艰辛的生活中，他和当地百姓渐渐地成为“一家人”。后来，当他离开时，他说，陕北的人民养育了我，保护了我。我虽然告别了陕北的父老兄弟，但再也离不开人民。1982年初，在北京工作的习近平，又主动要求“沉”到基层，来到了河北省正定县。抱着一颗为人民做事情的心，习近平把千千万万个家庭的美好生活作为自己的奋斗目标。这其中，7000多万贫困人口、数千万家庭能否如期脱贫，走上幸福之路，最让他揪心。党的十八大以来，他30余次到国内各地考察，有一半以上都涉及扶贫开发问题。每次考察扶贫，翻山越岭、风雪兼程，他都会走进一户户困难群众的家中，不懈的脚步只为丈量真实的贫困角落。他有时会盘腿上炕，拉着乡亲手详细询问他们一年下来有多少收入，粮食够不够吃，过冬的棉被有没有。他有时会掀开褥子看炕垒得好不好，问屋顶上铺没铺油毡、会不会漏雨。群众的困难他会放在心上，群众过上好日子他会喜在心里。在他的重视和推动下，从2012年到2014年，全国农村贫困人口减少5221万人。每减少一户贫困家庭，他都会喜在心里。
"""
  
  //Segmentor.parseTuple(text).foreach(println)
  ZhSegmentor().parseTuple(text).foreach(println)
	//println(Segmentor.newWordJava(text,10).asScala.toString)

	}
}
/**
 * 中文分词器。
 */
object Segmentor extends Serializable{
  /**默认的分词类型：Nlp分词，To精准分词，Base基本分词。*/
	var segType="Nlp"  //Nlp,To
	var learnTool = new LearnTool()
	var confPath=System.getProperty("user.dir")
	/*var readConfPath=System.getProperty("user.dir")  //"."
	var confPath=readConfPath  //"."
	try{
	  readConfPath=System.getProperty("Bigdata.appConfPath").trim
	}catch{
	  case e:Throwable => {
	    println("Segmentor读appConfPath异常！")
	    readConfPath="."
	  }
	}
	if(readConfPath.nonEmpty && !readConfPath.equals(".")) confPath=readConfPath
	println("confPath="+confPath)
	*/
	var dicPath=""
	var dicSeg=""
	var userDic=List[String]()
	init(confPath+"/conf/dictionary/","segment")
	
	def setPath(dicPth:String,dicSegPth:String)={
	 	dicPath=dicPth
  	dicSeg=dicSegPth
  	loadUserLibrary(dicPath+dicSeg)
  	loadAmbiguityLibrary(dicPath+"ambiguity.dic") 
	}
	
	def init(dicPth:String,dicSegPth:String)={
	  setPath(dicPth,dicSegPth)
	  val dicFiles=new File(dicPth+dicSegPth+"/newwords.dic")
	  //println("dicFiles="+dicFiles+","+dicFiles.isFile)
	  //println("dicPath+dicSeg="+dicPath+dicSeg)
  	if(dicFiles.isFile){
  		userDic=FileUtils.readLines(dicFiles).filter { x => x.length()>0&&(x.contains("\t")) }.map { x => x.split("\t")(0).trim }.toList
  	}
  	//println("userDic="+userDic.take(100).mkString("\n"))
	  println(parseTuple("中文自然语言分词器。").mkString(" | "))
	}
	
	def loadDicLibrary(dicPath:String,dicSeg:String)={
	  loadUserLibrary(dicPath+dicSeg)
  	loadAmbiguityLibrary(dicPath+"ambiguity.dic")
	}
	
	def getSegType()=segType
	
	def setSegType(segT:String)={
		segType=segT
	}
	
	def loadUserLibrary(dicPth:String)={
		MyStaticValue.userLibrary=dicPth
	}
	def loadAmbiguityLibrary(dicPth:String)={
		MyStaticValue.ambiguityLibrary=dicPth
	}
	
	def getFilesList(path:String):scala.List[String]={
		var inputFiles=scala.List[String]()
		val fl = new File(path)

		val fs = fl.list()
		for (x <- fs) {
			if (!x.startsWith(".")){
				val opFile = new File(path + x)
				if(opFile.isFile()){
					inputFiles=(path+x)::inputFiles
				}else if(opFile.isDirectory()){
					val opFs = opFile.list()
					for(y<-opFs){
						inputFiles=(path+x+"/"+y)::inputFiles
					}
				}
			}
		}
		inputFiles
	}
	
	def addDicLibrary(dicPth:String)={
		val dic=FileUtils.readLines(new File(dicPth))
		dic.foreach( x => {
			val line=x.split("\t")
			if(line.length>2) {
				UserDefineLibrary.insertWord(line(0).trim, line(1).trim, line(2).trim.toInt)
			}else if(line.length==1 || line.length==2){
				UserDefineLibrary.insertWord(line(0).trim, "userDefine", 10)
			}
		})
	}
	
	def parse(text:String,learn:Boolean=false):List[Term]={
		segType match{
			case "Base" => return BaseAnalysis.parse(text.replaceAll("[\\s\u3000]","")).toList
			case "To" => return ToAnalysis.parse(text.replaceAll("[\\s\u3000]","")).toList
			case "Nlp" => if(!learn) return NlpAnalysis.parse(text.replaceAll("[\\s\u3000]","")).toList else NlpAnalysis.parse(text.replaceAll("[\\s\u3000]",""),learnTool).toList
			case _ => List[Term]()
		}
	}
	
	def parseTuple(text:String,learn:Boolean=false):List[(String,String)]={
		segType match{
			case "Base" => return BaseAnalysis.parse(text.replaceAll("[\\s\u3000]","")).map { x => (x.getName.trim,x.getNatureStr.trim) }.toList
			case "To" => return ToAnalysis.parse(text.replaceAll("[\\s\u3000]","")).map { x => (x.getName.trim,x.getNatureStr.trim) }.toList
			case "Nlp" => if(!learn) return NlpAnalysis.parse(text.replaceAll("[\\s\u3000]","")).map { x => (x.getName.trim,x.getNatureStr.trim) }.toList else NlpAnalysis.parse(text.replaceAll("[\\s\u3000]",""),learnTool).map { x => (x.getName.trim,x.getNatureStr.trim) }.toList
			case _ => List[(String,String)]()
		}
	}
	
	def newWord(text:String,top:Int):List[(String,Double)]={
		learnTool=null
		learnTool=new LearnTool()
		//words=null
		var words=List[Term]()
		words=parse(text.replaceAll("[\\s\u3000]",""),true)
		val topNewWords=learnTool.getTopTree(top)
		if(topNewWords!=null&&topNewWords.length>0) topNewWords.map(x=> (x.getKey,x.getValue.toDouble)).filter(w => !(w._1.matches("[\\s\u3000]")) ).toList else List[(String,Double)](("",0.0))
	}
	
	def getKeyWord(text:String,top:Int):List[(String,Double,Double)]={
		val kw=new KeyWordComputer(top).computeArticleTfidf(text.replaceAll("[\\s\u3000]",""))
		if(kw!=null&&kw.nonEmpty) {
			val keys=kw.map { x => (x.getName,x.getScore,x.getFreq.toDouble) }.toList
			var (sum1,sum2) = keys.map { w => (w._2,w._3) }.reduce((u,v)=>(u._1+v._1,u._2+v._2))
			if(sum1<0.1) sum1=1.0
			if(sum2<0.1) sum2=1.0
			keys.map( w=>(w._1, w._2/sum1, w._3/sum2) ).toList
		} else List[(String,Double,Double)]()
		//new KeyWordComputer(top).computeArticleTfidf(text.replaceAll("[\\s\u3000]","")).map { x => (x.getName,x.getScore,x.getFreq.toDouble) }.toList
	}
	
	/** 找出文本的新词。
	 *  @param text String，文本。
	 *  @param top Int，文本新词个数。
	 *  @return java.util.List<java.util.Map.Entry<String,java.lang.Double>>，新词列表。Entry(String,Double)表示新词和新词分数。
	 */
	def newWordJava(text:String,top:Int):java.util.List[java.util.Map.Entry[String,java.lang.Double]]={
		//learnTool=null
		learnTool=new LearnTool()
		//words=null
		var words=List[Term]()
		words=parse(text.replaceAll("[\\s\u3000]",""),true)
		val topNewWords=learnTool.getTopTree(top)
		learnTool=null
		val hmap=new HashMap[String, java.lang.Double]()
		hmap.put("无", 0.0)
		if(topNewWords!=null&&topNewWords.nonEmpty) {
			topNewWords.filter(x=>{
				val s=x.getKey
				!s.matches("[\\s\u3000]+") && s.length()>0 && (!userDic.contains(s))
			}) 
		}else{
			new java.util.ArrayList[java.util.Map.Entry[String,java.lang.Double]](hmap.entrySet().take(1))
		}
	}
		    
	def parseJava(text:String,learn:Boolean=false):java.util.List[String]={
		segType match{
			case "Base" => return BaseAnalysis.parse(text.replaceAll("[\\s\u3000]","")).map { x => x.getName }.toList.asJava
			case "To" => return ToAnalysis.parse(text.replaceAll("[\\s\u3000]","")).map { x => x.getName }.toList.asJava
			case "Nlp" => if(!learn) return NlpAnalysis.parse(text.replaceAll("[\\s\u3000]","")).map { x => x.getName }.toList.asJava else NlpAnalysis.parse(text.replaceAll("[\\s\u3000]",""),learnTool).map { x => x.getName }.toList.asJava
			case _ => List[String]().asJava
		}
	}
}
