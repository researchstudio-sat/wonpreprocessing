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

import gate.util.GateException;
import org.apache.commons.cli.*;
import org.apache.commons.mail.util.MimeMessageParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.mail.MessagingException;
import javax.mail.internet.MimeMessage;
import java.io.*;
import java.nio.charset.Charset;

/**
 * Created by hfriedrich on 26.06.2014.
 *
 * Preprocess mail files to produce input for the Gate processing and the matching algorithm.
 *
 * Input parameters:
 * args[0] - Input mail folder: place the mail .eml files to be processed here
 * args[1] - Output folder: preprocesses mail files to extract the important content to this folder.
 * Furthermore creates a subfolder called "rescal" were the connections.txt file must be placed befire execution,
 * and then creates the rescal tensor data and header files in this directory.
 */
public class MailProcessing
{
  private static final Logger logger = LoggerFactory.getLogger(MailProcessing.class);
  private static final String GATE_APP_PATH = "src/main/resources/gate/application.xgapp";

  private static final String FROM_PREFIX = "From: ";
  private static final String TO_PREFIX = "To: ";
  private static final String DATE_PREFIX = "Date: ";
  private static final String SUBJECT_PREFIX = "Subject: ";
  private static final String CONTENT_PREFIX = "Content: ";

  public static void main(String[] args) {

    String input = null;
    String output = null;
    String workingFolder = null;
    String connections = null;
    String gateApp = null;
    boolean createContentSlice = false;
    boolean useStemming = false;

    // create Options object for command line input
    Options options = new Options();
    options.addOption("input", true, "input mail file folder");
    options.addOption("output", true, "output results folder");
    options.addOption("connections", true, "connections txt file");
    options.addOption("gateapp" ,true, "gate application path (to .xgapp)");
    options.addOption("content", false, "create a content slice in addition to the subject and need type slices");
    options.addOption("stemming" , false, "use stemming in preprocessing");

    CommandLineParser parser = new BasicParser();
    try {
      CommandLine cmd = parser.parse(options, args);
      input = cmd.getOptionValue("input");
      output = cmd.getOptionValue("output");
      connections = cmd.getOptionValue("connections");
      gateApp = cmd.getOptionValue("gateapp", GATE_APP_PATH);
      createContentSlice = cmd.hasOption("content");
      useStemming = cmd.hasOption("stemming");

      if (input == null || output == null) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.printHelp( "-input <folder> -output <folder>  ... other optional options", options );
      }

      workingFolder = output + "/preprocessed";
      MailProcessing.preprocessMails(input, workingFolder);
      GateRESCALProcessing rescal = new GateRESCALProcessing(gateApp, workingFolder, createContentSlice, useStemming);
      rescal.processFilesWithGate(workingFolder);
      if (connections != null) {
        rescal.addConnectionData(connections);
      }
      rescal.createRescalData(output);

    } catch (ParseException e) {
      logger.error(e.getMessage(), e);
    } catch (IOException e) {
      logger.error(e.getMessage(), e);
    } catch (GateException e) {
      logger.error(e.getMessage(), e);
    }
  }

  public static String cleanFileName(String filename) {
    String result = filename.replaceAll("[+?]","");
    result = result.replaceAll("\\p{C}", "");
    return result;
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
      Writer fw = null;

      try {
        fis = new FileInputStream(file);
        MimeMessage emailMessage = new MimeMessage(null, fis);
        MimeMessageParser parser = new MimeMessageParser(emailMessage);
        parser.parse();
        String content = null;
        if (parser.hasPlainContent()) {
          content = parser.getPlainContent();
          int endIndex = content.indexOf("-------------");
          if (endIndex != -1) {
            content = content.substring(0, endIndex);
          }
        } else {
          logger.warn("no plain content in file: {}, use HTML content", file);
          content = parser.getHtmlContent();
        }

        File outfile = new File(outputFolder + "/" + cleanFileName(file.getName()));
        logger.debug("writing output file: {}", outfile.getAbsolutePath());
        logger.debug("- mail subject: {}", parser.getSubject());
        FileOutputStream outputStream = new FileOutputStream(outfile);

        // Enforce UTF-8 when writing files. Non UTF-8 files will be reported.
        fw = new OutputStreamWriter(outputStream, Charset.forName("UTF-8"));

        fw.append(FROM_PREFIX + parser.getFrom() + "\n");
        fw.append(TO_PREFIX + parser.getTo() + "\n");
        fw.append(DATE_PREFIX + emailMessage.getSentDate() + "\n");
        fw.append(SUBJECT_PREFIX + parser.getSubject() + "\n");
        fw.append(CONTENT_PREFIX + /*parser.getPlainContent()*/content + "\n");

      } catch (MessagingException me) {
        logger.error("Error opening mail file: " + file.getAbsolutePath(), me);
      } catch (IOException ioe) {
        logger.error("Error writing file: " + file.getAbsolutePath(), ioe);
        System.err.println("Error writing file: " + file.getAbsolutePath());
      } catch (Exception e) {
        logger.error("Error parsing mail file: " + file.getAbsolutePath(), e);
      } finally {
        if (fis != null) fis.close();
        if (fw != null) fw.close();
      }
    }
  }

}
