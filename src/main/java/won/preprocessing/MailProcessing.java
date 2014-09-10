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
import org.apache.commons.mail.util.MimeMessageParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.mail.MessagingException;
import javax.mail.internet.MimeMessage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;

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
  private static final String GATE_APP_PATH = "resources/gate/application.xgapp";

  private static final String FROM_PREFIX = "From: ";
  private static final String TO_PREFIX = "To: ";
  private static final String DATE_PREFIX = "Date: ";
  private static final String SUBJECT_PREFIX = "Subject: ";
  private static final String CONTENT_PREFIX = "Content: ";

  public static void main(String[] args) {

    if (args.length < 2) {
      System.err.println("USAGE: java MailProcessing <input_directory> <output_directory>");
    } else try {
      MailProcessing.preprocessMails(args[0], args[1]);
      GateRESCALProcessing rescal = new GateRESCALProcessing(GATE_APP_PATH, args[1]);
      rescal.processFilesWithGate(args[1]);
      rescal.addConnectionData(args[1] + "/rescal/connections.txt");
      rescal.createRescalData(args[1] + "/rescal");

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
      FileWriter fw = null;

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
        fw = new FileWriter(outfile);

        fw.append(FROM_PREFIX + parser.getFrom() + "\n");
        fw.append(TO_PREFIX + parser.getTo() + "\n");
        fw.append(DATE_PREFIX + emailMessage.getSentDate() + "\n");
        fw.append(SUBJECT_PREFIX + parser.getSubject() + "\n");
        fw.append(CONTENT_PREFIX + /*parser.getPlainContent()*/content + "\n");

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

}
