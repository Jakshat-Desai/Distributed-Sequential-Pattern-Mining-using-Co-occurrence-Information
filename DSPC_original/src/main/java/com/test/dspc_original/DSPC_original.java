package com.test.dspc_original;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class DSPC_original {

    //minimum support count
    private final static int min_sup = 13000;

    //First MapReduce Phase begins here
    public static class MapPhase1 extends Mapper<LongWritable, Text, Text, Text> {

        //First Map phase helper functions

        //function to create CMAPi
        private HashMap<String,Set<String>> createLi(String value)
        {
            HashMap<String,Set<String>> Li = new HashMap<>();
            String[] tokenizer = value.split("-1");
            String itemsetTokenizer;
            int tokenNum=0;
            while (tokenNum<tokenizer.length) {
                itemsetTokenizer = tokenizer[tokenNum];
                String[] itemset = itemsetTokenizer.split(" ");
                for(int i=0;i<itemset.length;i++)
                {
                    String itemi = itemset[i];
                    if(itemi.isEmpty())
                    {
                        continue;
                    }
                    if(itemi.equals("-2"))
                    {
                        break;
                    }
                    for(int j=i+1;j<itemset.length;j++)
                    {
                        String itemj = itemset[j];
                        if(itemj.isEmpty())
                        {
                            continue;
                        }
                        if(itemj.equals("-2"))
                        {
                            break;
                        }
                        Set<String> CLitemi;
                        if(Li.containsKey(itemi))
                        {
                            CLitemi = Li.get(itemi);
                            CLitemi.add(itemj);
                            Li.replace(itemi,CLitemi);
                        }
                        else
                        {
                            CLitemi = new HashSet<String>();
                            CLitemi.add(itemj);
                            Li.put(itemi,CLitemi);
                        }
                    }
                }
                tokenNum++;
            }
            return Li;
        }

        //function to create CMAPs
        private HashMap<String,Set<String>> createLs(String value)
        {
            HashMap<String,Set<String>> Ls = new HashMap<>();
            String[] tokenizer = value.split("-1");
            String itemsetTokenizer;
            List<String[]> prevItemSets = new ArrayList<>();
            int tokenNum=0;
            while (tokenNum<tokenizer.length) {
                itemsetTokenizer = tokenizer[tokenNum];
                String[] itemset = itemsetTokenizer.split(" ");
                for(int k=0;k<itemset.length;k++)
                {
                    String itemk = itemset[k];
                    if(itemk.isEmpty())
                    {
                        continue;
                    }
                    if(itemk.equals("-2"))
                    {
                        break;
                    }
                    for(String[] prevItemset: prevItemSets)
                    {
                        for (int j = 0; j < prevItemset.length; j++)
                        {
                            String itemj = prevItemset[j];
                            if(itemj.isEmpty())
                            {
                                continue;
                            }
                            if(itemj.equals("-2"))
                            {
                                break;
                            }
                            Set<String> CLitemj;
                            if (Ls.containsKey(itemj))
                            {
                                CLitemj = Ls.get(itemj);
                                CLitemj.add(itemk);
                                Ls.replace(itemj, CLitemj);
                            }
                            else
                            {
                                CLitemj = new HashSet<>();
                                CLitemj.add(itemk);
                                Ls.put(itemj, CLitemj);
                            }
                        }
                    }
                }
                prevItemSets.add(itemset);
                tokenNum++;
            }
            return Ls;
        }

        //CMAP context write helper
        private void mapContextWrite(int k, HashMap<String,Set<String>> CMAP,Context context) throws IOException, InterruptedException {
            for(HashMap.Entry<String,Set<String>> entry: CMAP.entrySet())
            {
                String v = "";
                for(String i: entry.getValue())
                {
                    v = v+" "+i;
                }
                String kval = entry.getKey();
                String keyString = ((k==-1)?"s":"i")+kval;
                Text val = new Text(v);
                Text key = new Text(keyString);
                context.write(key,val);
            }
        }

        //Mapping function for first phase
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            HashMap<String,Set<String>> Li = createLi(line);
            HashMap<String,Set<String>> Ls = createLs(line);
            Text k = new Text("q"+key.toString());
            context.write(k,value);
            mapContextWrite(1,Li,context);
            mapContextWrite(-1,Ls,context);
        }
    }

    //First Reduce phase
    public static class ReducePhase1 extends Reducer<Text, Text, Text, Text> {

        //First Reduce phase helper functions
        private MultipleOutputs output;

        @Override
        public void setup(Context context)
        {
            output = new MultipleOutputs(context);
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException
        {
            output.close();
        }

        //function to process and print CMAP data
        private void handleCMAP(char type, Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            HashMap<String,Integer> CMAP = new HashMap<>();
            for (Text val : values) {
                String[] temp = val.toString().split(" ");
                for (int c=0;c<temp.length;c++)
                {
                    if(temp[c].isEmpty())
                    {
                        continue;
                    }
                    String i = temp[c];
                    if(CMAP.containsKey(i))
                    {
                        CMAP.replace(i,CMAP.get(i)+1);
                    }
                    else
                    {
                        CMAP.put(i,1);
                    }
                }
            }
            Text val;
            String fileName = "CMAP";
            key.set(type+key.toString());
            for(HashMap.Entry<String,Integer> entry: CMAP.entrySet())
            {
                String t = entry.getKey();
                val = new Text(t);
                if(entry.getValue()>=min_sup)
                {
                    output.write(fileName,key,val);
                }
            }
        }


        //function to process and print sequence data
        private void handleSequences(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for(Text val: values)
            {
                Text v;
                String value = "";
                Integer i=1;
                String[] tokenizer = val.toString().split("-1");
                int tokenNum=0;
                while(tokenNum<tokenizer.length)
                {
                    String[] temp = tokenizer[tokenNum].split(" ");
                    for (int j = 0; j < temp.length; j++)
                    {
                        if(temp[j].isEmpty())
                        {
                            continue;
                        }
                        if(temp[j].equals("-2"))
                        {
                            break;
                        }
                        //item:position
                        value+=temp[j]+" "+i+" ";
                    }
                    i++;
                    tokenNum++;
                }
                v = new Text(value);
                output.write("freqItemPos",key,v);
            }
        }

        //Reduce function for first phase
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            char type = (char) key.charAt(0);
            String k = key.toString().substring(1);
            switch (type)
            {
                case 's':
                case 'i':
                    handleCMAP(type,new Text(k),values,context);
                    break;
                case 'q':
                    handleSequences(new Text(k),values,context);
            }
        }

    }

    //Second MapReduce Phase Begins here

    //Second Map phase
    public static class MapPhase2 extends Mapper<LongWritable, Text, Text, IntWritable>
    {
        private HashMap<String,Set<String>> CMAPi;
        private HashMap<String,Set<String>> CMAPs;

        //function to read CMAPi and CMAPs from distributed cache
        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            CMAPi = new HashMap<>();
            CMAPs = new HashMap<>();
            URI[] cacheFiles = context.getCacheFiles();

            for (int i = 0; cacheFiles != null && i < cacheFiles.length; i++)
            {
                try
                {
                    String line = "";
                    FileSystem fs = FileSystem.get(context.getConfiguration());
                    String path = cacheFiles.toString();
                    Path getFilePath = new Path(cacheFiles[i].toString());
                    String temp[] = path.split("/");
                    char type;
                    BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(getFilePath)));
                    while ((line = reader.readLine()) != null) {
                        type = line.charAt(0);
                        line = line.substring(1);
                        StringTokenizer tokenizer = new StringTokenizer(line);
                        String item = tokenizer.nextToken();
                        String val = tokenizer.nextToken();
                        Set<String> CL;
                        if (type == 'i')
                        {
                            if (CMAPi.containsKey(item))
                            {
                                CL = CMAPi.get(item);
                                CL.add(val);
                                CMAPi.replace(item, CL);
                            }
                            else
                            {
                                CL = new HashSet<>();
                                CL.add(val);
                                CMAPi.put(item, CL);
                            }
                        }
                        else
                        {
                            if (CMAPs.containsKey(item))
                            {
                                CL = CMAPs.get(item);
                                CL.add(val);
                                CMAPs.replace(item, CL);
                            }
                            else
                            {
                                CL = new HashSet<>();
                                CL.add(val);
                                CMAPs.put(item, CL);
                            }
                        }
                    }
                } catch (Exception e) {
                    System.out.println("Unable to read the File");
                    System.exit(1);
                }
            }
        }

        //helper functions for second map phase

        private HashMap<String,HashMap<Integer,Set<Integer>>> createSIL(Integer Sid, HashMap<String,Set<Integer>> itemPosMapping)
        {
            HashMap<String,HashMap<Integer,Set<Integer>>> SIL = new HashMap<>();
            for(HashMap.Entry<String,Set<Integer>> entry: itemPosMapping.entrySet())
            {
                Set<Integer> itemPosList = entry.getValue();
                String item = entry.getKey()+" -1 -2";
                HashMap<Integer,Set<Integer>> data = new HashMap<>();
                data.put(Sid,itemPosList);
                SIL.put(item,data);
            }
            return SIL;
        }

        private boolean earlyPrune(String Seq, String lastitem)
        {
            String[] itemsets = Seq.split("-1");
            String[] ik = itemsets[itemsets.length-2].split(" ");
            String[] ik_1 = ((itemsets.length>=3)?itemsets[itemsets.length-3]:"").split(" ");
            for(int i=0;i<ik.length;i++)
            {
                String x = ik[i];
                if(x.isEmpty() || x.equals(lastitem))
                {
                    continue;
                }
                if(!CMAPi.containsKey(x) || !CMAPi.get(x).contains(lastitem))
                {
                    return true;
                }
            }
            for(int i=0;i<ik_1.length;i++)
            {
                String y = ik_1[i];
                if(y.isEmpty())
                {
                    continue;
                }
                if(!CMAPs.containsKey(y) || !CMAPs.get(y).contains(lastitem))
                {
                    return true;
                }
            }
            return false;
        }

        private HashMap<Integer,Set<Integer>> createSILsequence(boolean itemEx, HashMap<Integer,Set<Integer>> X,HashMap<Integer,Set<Integer>> Y)
        {
            HashMap<Integer,Set<Integer>> Z = new HashMap<>();
            for(HashMap.Entry<Integer,Set<Integer>> entry: X.entrySet())
            {
                Set<Integer> polistx = entry.getValue();
                Set<Integer> polistz = new HashSet<>();
                for(HashMap.Entry<Integer,Set<Integer>> entry1: Y.entrySet())
                {
                    if(entry.getKey()!=entry1.getKey())
                    {
                        continue;
                    }
                    Set<Integer> polisty = entry1.getValue();
                    if(itemEx)
                    {
                        for(Integer posx: polistx)
                        {
                            for(Integer posy: polisty)
                            {
                                if(posx==posy)
                                {
                                    polistz.add(posx);
                                    break;
                                }
                            }
                        }
                    }
                    else
                    {
                        for(Integer posx: polistx)
                        {
                            for(Integer posy: polisty)
                            {
                                if(posy>posx)
                                {
                                    polistz.add(posy);
                                }
                            }
                        }
                    }
                }
                if(polistz.size()>0)
                {
                    Z.put(entry.getKey(), polistz);
                }
            }
            return Z;
        }

        private void generateSequences(String Seq, HashMap<String,HashMap<Integer,Set<Integer>>> SIL, Context context) throws IOException, InterruptedException
        {
            if(Seq.length()<=6)
            {
                return;
            }
            String lastitem = "";
            for(int i=Seq.length()-7;i>=0 && Seq.charAt(i)!=' ';i--)
            {
                lastitem=Seq.charAt(i)+lastitem;
            }
            String newSeq;
            if(CMAPs.containsKey(lastitem))
            {
                for (String itemc : CMAPs.get(lastitem))
                {
                    if(!SIL.containsKey(itemc+" -1 -2"))
                    {
                        continue;
                    }
                    newSeq=Seq.substring(0,Seq.length()-2)+itemc+" -1 -2";
                    if(!earlyPrune(newSeq,itemc))
                    {
                        HashMap<Integer,Set<Integer>> SILnewSeq = createSILsequence(false,SIL.get(Seq),SIL.get(itemc+" -1 -2"));
                        SIL.put(newSeq,SILnewSeq);
                        Integer count = SILnewSeq.size();
                        if(count>=1)
                        {
                            context.write(new Text(newSeq+"\t"),new IntWritable(count));
                            generateSequences(newSeq,SIL,context);
                        }
                    }
                }
            }
            if(CMAPi.containsKey(lastitem))
            {
                for(String itemc: CMAPi.get(lastitem))
                {
                    if(!SIL.containsKey(itemc+" -1 -2"))
                    {
                        continue;
                    }
                    newSeq=Seq.substring(0,Seq.length()-5)+itemc+" -1 -2";
                    if(!earlyPrune(newSeq,itemc))
                    {
                        HashMap<Integer,Set<Integer>> SILnewSeq = createSILsequence(true,SIL.get(Seq),SIL.get(itemc+" -1 -2"));
                        SIL.put(newSeq,SILnewSeq);
                        Integer count = SILnewSeq.size();
                        if(count>=1)
                        {
                            context.write(new Text(newSeq+"\t"),new IntWritable(count));
                            generateSequences(newSeq,SIL,context);
                        }
                    }
                }
            }
        }

        //Mapping function for second phase
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException
        {

            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line);
            Integer Sid = Integer.parseInt(tokenizer.nextToken());
            HashMap<String,Set<Integer>> itemPosMapping = new HashMap<>();
            while (tokenizer.hasMoreTokens())
            {
                String item = tokenizer.nextToken();
                Integer position = Integer.parseInt(tokenizer.nextToken());
                if(itemPosMapping.containsKey(item))
                {
                    Set<Integer> itemsetPosList = itemPosMapping.get(item);
                    itemsetPosList.add(position);
                    itemPosMapping.replace(item,itemsetPosList);
                }
                else
                {
                    Set<Integer> itemsetPosList = new HashSet<>();
                    itemsetPosList.add(position);
                    itemPosMapping.put(item,itemsetPosList);
                }
            }
            HashMap<String,HashMap<Integer,Set<Integer>>> SIL = createSIL(Sid,itemPosMapping);
            HashMap<String,HashMap<Integer,Set<Integer>>> SILcpy = new HashMap<>();
            for(HashMap.Entry<String,HashMap<Integer,Set<Integer>>> entry: SIL.entrySet())
            {
                SILcpy.put(entry.getKey(),entry.getValue());
            }
            for(HashMap.Entry<String,HashMap<Integer,Set<Integer>>> entry: SILcpy.entrySet())
            {
                String k = entry.getKey();
                generateSequences(k, SIL, context);
            }
        }
    }

    //Second Reduce phase
    public static class ReducePhase2 extends Reducer<Text, IntWritable, Text, IntWritable>
    {
        //Reduce function for second phase
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException
        {
            int tot_count = 0;
            for(IntWritable value : values)
            {
                tot_count += value.get();
            }
            if(tot_count>=min_sup)
            {
                context.write(key, new IntWritable(tot_count));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        Job job1 = new Job(conf, "Phase1");

        job1.setJarByClass(DSPC_original.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);

        job1.setMapperClass(MapPhase1.class);
        job1.setReducerClass(ReducePhase1.class);

        job1.setInputFormatClass(TextInputFormat.class);
        job1.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job1, new Path(args[0]));
        MultipleOutputs.addNamedOutput(job1, "CMAP", TextOutputFormat.class, Text.class, Text.class );
        MultipleOutputs.addNamedOutput(job1, "freqItemPos", TextOutputFormat.class, Text.class, Text.class );
        FileOutputFormat.setOutputPath(job1, new Path(args[1]+"/first_phase_output"));

        job1.waitForCompletion(true);

        Job job2 = new Job(conf,"Phase2");

        try {
            job2.addCacheFile(new URI("hdfs://namenode:9820"+args[1]+"/first_phase_output/CMAP-r-00000"));
        }
        catch (Exception e) {
            System.out.println("CMAP File Not Added");
            System.exit(1);
        }

        job2.setJarByClass(DSPC_original.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);

        job2.setMapperClass(MapPhase2.class);
        job2.setReducerClass(ReducePhase2.class);

        job2.setInputFormatClass(TextInputFormat.class);
        job2.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job2, new Path(args[1]+"/first_phase_output/freqItemPos-r-00000"));
        FileOutputFormat.setOutputPath(job2, new Path(args[1]+"/second_phase_output"));

        job2.waitForCompletion(true);
    }

}
