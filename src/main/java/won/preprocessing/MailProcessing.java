/*
 * Copyright 2012  Research Studios Austria Forschungsges.m.b.H.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package won.preprocessing;

import gate.Corpus;
import gate.Factory;
import gate.Gate;
import gate.creole.SerialAnalyserController;
import gate.util.GateException;
import org.apache.commons.mail.util.MimeMessageParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.mail.MessagingException;
import javax.mail.internet.MimeMessage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.net.MalformedURLException;

/**
 * Created by hfriedrich on 26.06.2014.
 */
public class MailProcessing
{
  private static final Logger logger = LoggerFactory.getLogger(MailProcessing.class);

  private static final String[] gateProcessingResources = {"gate.creole.tokeniser.DefaultTokeniser",
                                                           "gate.creole.splitter.SentenceSplitter"};

  public static void main(String[] args) {

    if (args.length < 2) {
      System.err.println("USAGE: java MailProcessing <input_directory> <output_directory>");
    } else try {
      MailProcessing.preprocessMails(args[0], args[1]);
      MailProcessing.processFilesWithGate(args[1]);
    } catch (IOException e) {
      e.printStackTrace();
    } catch (GateException e) {
      e.printStackTrace();
    }
  }

  /**
   * Read mail files from the input folder, extract several fields (e.g. subject, content, from,
   * to) and save this data back into a text file of the output folder.
   *
   * @param inputFolder  input folder with the mails
   * @param outputFolder output folder with extracted content files
   * @throws IOException
   */
  private static void preprocessMails(String inputFolder, String outputFolder) throws IOException {

    File inFolder = new File(inputFolder);
    File outFolder = new File(outputFolder);
    outFolder.mkdirs();

    if (!inFolder.isDirectory()) {
      throw new IOException("Input folder not a directory: " + inputFolder);
    }
    if (!outFolder.isDirectory()) {
      throw new IOException("Output folder not a directory: " + outputFolder);
    }

    logger.info("preprocessing mail files: ");
    logger.info("- input folder {}", inputFolder);
    logger.info("- output folder {}", outputFolder);

    for (File file : inFolder.listFiles()) {
      if (file.isDirectory()) {
        continue;
      }

      logger.debug("processing mail file: {} ", file);
      FileInputStream fis = null;
      FileWriter fw = null;

      try {
        fis = new FileInputStream(file);
        MimeMessage emailMessage = new MimeMessage(null, fis);
        MimeMessageParser parser = new MimeMessageParser(emailMessage);
        parser.parse();
        String content = null;
        if (parser.hasPlainContent()) {
          content = parser.getPlainContent();
        } else {
          logger.warn("no plain content in file");
          continue;
        }

        File outfile = new File(outputFolder + "/" + file.getName());
        logger.debug("writing output file: {}", outfile.getAbsolutePath());
        logger.debug("- mail subject: {}", parser.getSubject());
        fw = new FileWriter(outfile);

        fw.append("From: " + parser.getFrom() + "\n");
        fw.append("To: " + parser.getTo() + "\n");
        fw.append("SentDate: " + emailMessage.getSentDate() + "\n");
        fw.append("ReceivedDate: " + emailMessage.getReceivedDate() + "\n");
        fw.append("Subject: " + parser.getSubject() + "\n");
        fw.append("Content: " + parser.getPlainContent() + "\n");

      } catch (MessagingException me) {
        logger.error("Error opening mail file: " + file.getAbsolutePath(), me);
      } catch (Exception e) {
        logger.error("Error parsing mail file: " + file.getAbsolutePath(), e);
      } finally {
        if (fis != null) fis.close();
        if (fw != null) fw.close();
      }
    }
  }

  /**
   * After the mails have been preprocessed by {@link #preprocessMails(String,
   * String)} the Gate processing can be applied.
   *
   * @param folder corpus document input folder
   * @throws GateException
   * @throws MalformedURLException
   */
  private static void processFilesWithGate(String folder) throws GateException, MalformedURLException {

    // init Gate
    Gate.init();
    Gate.getCreoleRegister().registerDirectories(
      new File(System.getProperty("user.dir")).toURL());

    // add files to a corpus
    File corpusFolder = new File(folder);
    Corpus corpus = Factory.newCorpus("Transient Gate Corpus");
    for (File file : corpusFolder.listFiles()) {
      corpus.add(Factory.newDocument(file.getAbsolutePath()));
    }

    // create the gate pipeline
    SerialAnalyserController pipeline = (SerialAnalyserController) Factory
      .createResource("gate.creole.SerialAnalyserController");
    pipeline.setCorpus(corpus);
    for (String pr : gateProcessingResources) {
      pipeline.add((gate.LanguageAnalyser) Factory.createResource(pr));
    }

    // process the documents using Gate
    logger.info("processing files with gate in folder: {}", folder);
    pipeline.execute();
  }

}
