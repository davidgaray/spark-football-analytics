import org.apache.spark.sql
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType

case class Game(GameID: String, date: String, home: String, away: String, homescore: Int, awayscore: Int, Season: Int) 
val gamesSchema = ScalaReflection.schemaFor[Game].dataType.asInstanceOf[StructType]

var df = spark.read.format("csv").option("header", "true").load("./nflscrapR/data/season_play_by_play/pbp_*.csv")
var gamesDf = spark.read.format("csv").option("header", "true").schema(gamesSchema).load("./nflscrapR/data/season_games/games_*.csv").as[Game].toDF

// plays in the first quarter
//df.filter($"qtr".isNotNull).filter(df("qtr").gt(0)).filter(df("qtr").equalTo(1)).filter($"OffenseTeam".isNotNull).groupBy("OffenseTeam").count().sort($"count".desc).show(32)
//df.filter($"qtr".isNotNull).filter(df("qtr").gt(0)).filter(df("qtr").equalTo(1)).filter($"OffenseTeam".isNotNull).withColumn("play_count", lit(1)).filter(df("PlayType").equalTo("RUSH")).groupBy("OffenseTeam").count().sort($"count".desc).show(32)
// all plays, order by percent rushing
// df.filter($"qtr".isNotNull).filter(df("qtr").lt(3)).filter($"OffenseTeam".isNotNull).withColumn("play_count", lit(1)).withColumn("rush_count", when($"PlayType".equalTo("RUSH"), 1).otherwise(0)).groupBy("OffenseTeam").agg(sum("play_count").as("play_count"), sum("rush_count").as("rush_count")).withColumn("percent_rush", col("rush_count") / col("play_count")).sort($"percent_rush".desc).show(32)

// is this casting necessary?
// df = df.withColumn("Minute", $"Minute".cast(sql.types.IntegerType)).withColumn("Second", $"Second".cast(sql.types.IntegerType))

// touchdowns
//df.filter(df("GameID").equalTo("2017090700")).filter(df("Touchdown") === 1).filter($"sp" === 1).sort($"qtr", $"Time".desc).select("qtr", "Time", "Touchdown", "sp")

//scoring plays
//df.filter(df("GameID").equalTo("2017090700")).filter($"sp" === 1).sort($"qtr", $"Time".desc).select("qtr", "Time", "Touchdown", "sp", "ScoreDiff", "desc").show(50, false)

//var pctRunningDf = df.filter($"qtr".isNotNull).filter(df("qtr").gt(2)).filter($"OffenseTeam".isNotNull).withColumn("play_count", lit(1)).withColumn("rush_count", when($"PlayType".equalTo("RUSH"), 1).otherwise(0)).groupBy("OffenseTeam").agg(sum("play_count").as("play_count"), sum("rush_count").as("rush_count")).withColumn("percent_rush", col("rush_count") / col("play_count")).sort($"percent_rush".desc)

gamesDf = gamesDf.withColumn("winner", when($"homescore" > $"awayscore", $"home").when($"homescore" < $"awayscore", $"away").otherwise("TIE"))
var seasonWinsDf = gamesDf.groupBy("Season", "winner").agg(count("*").alias("wins")).withColumnRenamed("winner", "team")
//seasonWinsDf.filter($"team" === "SEA").orderBy("Season").show
var winsJoinDf = seasonWinsDf.withColumnRenamed("team", "posteam")

//TODO: filter fourth quarter
var pctRunningDf = df.filter($"qtr".isNotNull).filter(df("qtr").gt(0)).filter($"posteam".isNotNull).withColumn("play_count", when($"PlayType".equalTo("Run") || $"PlayType".equalTo("Pass"), 1).otherwise(0)).withColumn("rush_count", when($"PlayType".equalTo("Run"), 1).otherwise(0)).groupBy("posteam", "Season").agg(sum("play_count").as("play_count"), sum("rush_count").as("rush_count")).withColumn("percent_rush", col("rush_count") / col("play_count")).sort($"percent_rush".desc)
pctRunningDf = pctRunningDf.join(winsJoinDf, Seq("posteam", "Season"))

var closeGamePctRunningDf = df.filter($"AbsScoreDiff".lt(8)).filter($"qtr".isNotNull).filter(df("qtr").gt(0)).filter($"posteam".isNotNull).withColumn("play_count", when($"PlayType".equalTo("Run") || $"PlayType".equalTo("Pass"), 1).otherwise(0)).withColumn("rush_count", when($"PlayType".equalTo("Run"), 1).otherwise(0)).groupBy("posteam", "Season").agg(sum("play_count").as("play_count"), sum("rush_count").as("rush_count")).withColumn("percent_rush", col("rush_count") / col("play_count")).sort($"percent_rush".desc)
closeGamePctRunningDf = closeGamePctRunningDf.join(winsJoinDf, Seq("posteam", "Season"))

var losingBySevenOrLessPctRunningDf = df.filter($"ScoreDiff".gt(-8)).filter($"ScoreDiff".lt(0)).filter($"qtr".isNotNull).filter(df("qtr").gt(0)).filter($"posteam".isNotNull).withColumn("play_count", when($"PlayType".equalTo("Run") || $"PlayType".equalTo("Pass"), 1).otherwise(0)).withColumn("rush_count", when($"PlayType".equalTo("Run"), 1).otherwise(0)).groupBy("posteam", "Season").agg(sum("play_count").as("play_count"), sum("rush_count").as("rush_count")).withColumn("percent_rush", col("rush_count") / col("play_count")).sort($"percent_rush".desc)
losingBySevenOrLessPctRunningDf = losingBySevenOrLessPctRunningDf.join(winsJoinDf, Seq("posteam", "Season"))
losingBySevenOrLessPctRunningDf.show(400)

import org.apache.spark.ml.feature.QuantileDiscretizer
var binner = new QuantileDiscretizer().setInputCol("percent_rush").setOutputCol("percent_rush_bin").setNumBuckets(10)
var targetDf = closeGamePctRunningDf
targetDf = binner.fit(targetDf).transform(targetDf)
targetDf.groupBy("percent_rush_bin").agg(avg("percent_rush").as("avg_percent_rush"), avg("wins").as("avg_wins")).orderBy("percent_rush_bin").show
//pctRunningDf.show(200)
//closeGamePctRunningDf.show(200)

